from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .rl_env import Go2StandBalanceEnv

try:
    import gymnasium as gym
except ImportError:
    gym = None


class AmpDiscriminator:
    def __init__(
        self,
        *,
        reference_transitions: np.ndarray,
        input_dim: int,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        buffer_size: int = 200_000,
    ) -> None:
        try:
            import torch as th
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is missing. Install training dependencies with: pip install -e '.[train]'"
            ) from exc

        self.th = th
        self.device = th.device(device if device != "auto" else "cpu")
        reference_tensor = th.as_tensor(reference_transitions, dtype=th.float32, device=self.device)
        self.feature_mean = reference_tensor.mean(dim=0, keepdim=True)
        self.feature_std = reference_tensor.std(dim=0, keepdim=True).clamp_min(1e-3)
        self.reference_transitions = self._normalize(reference_tensor)
        self.input_dim = int(input_dim)
        self.policy_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(buffer_size)))
        self.net = th.nn.Sequential(
            th.nn.Linear(self.input_dim, 256),
            th.nn.ELU(),
            th.nn.Linear(256, 128),
            th.nn.ELU(),
            th.nn.Linear(128, 1),
        ).to(self.device)
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=learning_rate)
        self._rng = np.random.default_rng()

    def add_policy_transition(self, transition: np.ndarray) -> None:
        if transition.shape != (self.input_dim,):
            return
        self.policy_buffer.append(transition.astype(np.float32, copy=True))

    def reward(self, transition: np.ndarray) -> float:
        if transition.shape != (self.input_dim,):
            return 0.0
        with self.th.no_grad():
            x = self.th.as_tensor(transition, dtype=self.th.float32, device=self.device).reshape(1, -1)
            x = self._normalize(x)
            logit = self.net(x)
            style = -self.th.nn.functional.logsigmoid(-logit)
            return float(self.th.clamp(style, min=0.0, max=5.0).item())

    def train_step(self, *, batch_size: int = 256, gradient_penalty: float = 0.0) -> dict[str, float]:
        if not self.policy_buffer:
            return {"amp/discriminator_loss": 0.0, "amp/reference_acc": 0.0, "amp/policy_acc": 0.0}

        th = self.th
        batch_size = max(2, int(batch_size))
        half = max(1, batch_size // 2)
        real_idx = th.randint(0, len(self.reference_transitions), (half,), device=self.device)
        real = self.reference_transitions[real_idx]

        fake_np = np.asarray(
            [self.policy_buffer[index] for index in self._rng.integers(0, len(self.policy_buffer), size=half)],
            dtype=np.float32,
        )
        fake = self._normalize(th.as_tensor(fake_np, dtype=th.float32, device=self.device))

        real_logits = self.net(real)
        fake_logits = self.net(fake)
        real_loss = th.nn.functional.binary_cross_entropy_with_logits(real_logits, th.ones_like(real_logits))
        fake_loss = th.nn.functional.binary_cross_entropy_with_logits(fake_logits, th.zeros_like(fake_logits))
        loss = real_loss + fake_loss

        if gradient_penalty > 0:
            real_for_grad = real.detach().requires_grad_(True)
            real_grad_logits = self.net(real_for_grad)
            gradients = th.autograd.grad(
                outputs=real_grad_logits.sum(),
                inputs=real_for_grad,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            loss = loss + gradient_penalty * th.mean(th.sum(gradients * gradients, dim=1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with th.no_grad():
            real_acc = th.mean((real_logits > 0).float())
            fake_acc = th.mean((fake_logits < 0).float())
        return {
            "amp/discriminator_loss": float(loss.item()),
            "amp/reference_acc": float(real_acc.item()),
            "amp/policy_acc": float(fake_acc.item()),
            "amp/policy_buffer": float(len(self.policy_buffer)),
        }

    def _normalize(self, transitions: Any) -> Any:
        return (transitions - self.feature_mean) / self.feature_std


class AmpRewardWrapper(gym.Wrapper if gym is not None else object):
    def __init__(
        self,
        env: Any,
        *,
        discriminator: AmpDiscriminator,
        style_weight: float,
        task_weight: float = 1.0,
    ) -> None:
        if gym is None:
            raise RuntimeError(
                "Gymnasium is missing. Install training dependencies with: pip install -e '.[train]'"
            )
        super().__init__(env)
        self.discriminator = discriminator
        self.style_weight = float(style_weight)
        self.task_weight = float(task_weight)
        self._previous_amp_observation: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._previous_amp_observation = _amp_from_info(info)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, task_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        current = _amp_from_info(info)
        style_reward = 0.0
        if self._previous_amp_observation is not None and current is not None:
            transition = np.concatenate([self._previous_amp_observation, current]).astype(np.float32)
            self.discriminator.add_policy_transition(transition)
            style_reward = self.discriminator.reward(transition)
        combined_reward = self.task_weight * float(task_reward) + self.style_weight * style_reward
        info["task_reward"] = float(task_reward)
        info["amp_style_reward"] = float(style_reward)
        info["amp_weighted_style_reward"] = float(self.style_weight * style_reward)
        info.setdefault("reward_terms", {})["amp_style"] = float(self.style_weight * style_reward)
        self._previous_amp_observation = current
        return obs, combined_reward, terminated, truncated, info


class AmpLoggingCallback:
    def __init__(
        self,
        *,
        discriminator: AmpDiscriminator,
        train_freq: int = 1024,
        batch_size: int = 256,
        updates: int = 1,
        gradient_penalty: float = 0.0,
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError(
                "Stable-Baselines3 is missing. Install training dependencies with: "
                "pip install -e '.[train]'"
            ) from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: AmpLoggingCallback) -> None:
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer._on_step(self)

        self._callback = _Callback(self)
        self.discriminator = discriminator
        self.train_freq = max(1, int(train_freq))
        self.batch_size = max(2, int(batch_size))
        self.updates = max(1, int(updates))
        self.gradient_penalty = max(0.0, float(gradient_penalty))

    @property
    def callback(self) -> Any:
        return self._callback

    def _on_step(self, callback: Any) -> bool:
        infos = callback.locals.get("infos", [])
        if infos:
            mean_style = float(np.mean([float(info.get("amp_style_reward", 0.0)) for info in infos]))
            mean_weighted = float(np.mean([float(info.get("amp_weighted_style_reward", 0.0)) for info in infos]))
            callback.logger.record("amp/style_reward", mean_style)
            callback.logger.record("amp/weighted_style_reward", mean_weighted)

        if callback.n_calls % self.train_freq == 0:
            report: dict[str, float] = {}
            for _ in range(self.updates):
                report = self.discriminator.train_step(
                    batch_size=self.batch_size,
                    gradient_penalty=self.gradient_penalty,
                )
            for key, value in report.items():
                callback.logger.record(key, value)
        return True


def load_amp_reference_transitions(
    paths: Iterable[str | Path],
    *,
    env: Go2StandBalanceEnv,
) -> np.ndarray:
    transitions: list[np.ndarray] = []
    for path in paths:
        if Path(path).suffix == ".npz":
            transitions.extend(_transitions_from_npz_reference(path))
            continue
        features = _features_from_trajectory(path, env=env)
        transitions.extend(
            np.concatenate([features[index], features[index + 1]]).astype(np.float32)
            for index in range(len(features) - 1)
        )

    if not transitions:
        raise ValueError("AMP reference needs at least two frames across the provided reference files")
    return np.asarray(transitions, dtype=np.float32)


def _transitions_from_npz_reference(path: str | Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "amp_observations" not in data:
            raise ValueError(f"NPZ AMP reference missing amp_observations: {path}")
        features = np.asarray(data["amp_observations"], dtype=np.float32)
        terminals = np.asarray(data["terminals"], dtype=np.bool_) if "terminals" in data else np.zeros(len(features), dtype=np.bool_)

    transitions: list[np.ndarray] = []
    for index in range(len(features) - 1):
        if bool(terminals[index]):
            continue
        transitions.append(np.concatenate([features[index], features[index + 1]]).astype(np.float32))
    return transitions


def _features_from_trajectory(path: str | Path, *, env: Go2StandBalanceEnv) -> list[np.ndarray]:
    trajectory = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(trajectory.get("Frames"), list):
        return _features_from_motion_imitation(trajectory, env=env)

    frames = trajectory.get("frames") or []
    if len(frames) < 2:
        return []

    joint_names = list(env.joint_names)
    default_targets = env.default_targets
    result: list[np.ndarray] = []
    previous_joint_values: dict[str, float] | None = None
    previous_t: float | None = None

    for frame in frames:
        base = frame.get("base") or {}
        position = list(base.get("position") or [0.0, 0.0, env.config.target_base_z])
        quaternion = list(base.get("quaternion") or [1.0, 0.0, 0.0, 0.0])
        linear_velocity = list(base.get("linear_velocity") or [0.0, 0.0, 0.0])[:3]
        angular_velocity = list(base.get("angular_velocity") or [0.0, 0.0, 0.0])[:3]
        joints = frame.get("joints") or {}
        t = float(frame.get("t", 0.0))
        joint_values = {name: float(joints.get(name, default_targets[name])) for name in joint_names}
        dt = max(t - previous_t, 1e-6) if previous_t is not None else 1e-6
        joint_vel = [
            (joint_values[name] - previous_joint_values[name]) / dt
            if previous_joint_values is not None
            else 0.0
            for name in joint_names
        ]
        joint_pos_rel = [joint_values[name] - default_targets[name] for name in joint_names]
        result.append(
            np.asarray(
                [
                    float((position + [env.config.target_base_z, env.config.target_base_z, env.config.target_base_z])[2]),
                    *_projected_gravity(quaternion),
                    *[float(v) for v in (linear_velocity + [0.0, 0.0, 0.0])[:3]],
                    *[float(v) for v in (angular_velocity + [0.0, 0.0, 0.0])[:3]],
                    *joint_pos_rel,
                    *joint_vel,
                ],
                dtype=np.float32,
            )
        )
        previous_joint_values = joint_values
        previous_t = t
    return result


def _features_from_motion_imitation(motion: dict[str, Any], *, env: Go2StandBalanceEnv) -> list[np.ndarray]:
    frames = motion.get("Frames") or []
    if len(frames) < 2:
        return []

    frame_duration = float(motion.get("FrameDuration") or 0.01667)
    joint_names = list(env.joint_names)
    default_targets = env.default_targets
    result: list[np.ndarray] = []
    previous_joint_values: dict[str, float] | None = None

    for index, frame in enumerate(frames):
        values = [float(value) for value in frame]
        if len(values) < 19:
            continue
        position = values[0:3]
        raw_joints = values[7:19]
        joint_values = {
            name: float(raw_joints[joint_index])
            for joint_index, name in enumerate(joint_names)
        }
        dt = frame_duration if index > 0 else 1e-6
        linear_velocity = [
            float(env.config.target_forward_velocity),
            float(env.config.target_lateral_velocity),
            0.0,
        ]
        joint_vel = [
            (joint_values[name] - previous_joint_values[name]) / dt
            if previous_joint_values is not None
            else 0.0
            for name in joint_names
        ]
        joint_pos_rel = [joint_values[name] - default_targets[name] for name in joint_names]
        result.append(
            np.asarray(
                [
                    position[2],
                    0.0,
                    0.0,
                    -1.0,
                    *linear_velocity,
                    0.0,
                    0.0,
                    0.0,
                    *joint_pos_rel,
                    *joint_vel,
                ],
                dtype=np.float32,
            )
        )
        previous_joint_values = joint_values

    return result


def _amp_from_info(info: dict[str, Any]) -> np.ndarray | None:
    values = info.get("amp_observation")
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def _projected_gravity(quaternion: list[float]) -> list[float]:
    quat = np.asarray((quaternion + [1.0, 0.0, 0.0, 0.0])[:4], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / norm
    w, x, y, z = quat
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return [float(value) for value in gravity]
