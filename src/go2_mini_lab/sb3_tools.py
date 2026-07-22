from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .diagnostics import ACTION_BALANCE_KEYS, CONTACT_BALANCE_KEYS, action_balance_metrics, contact_balance_metrics
from .force_metrics import ForceComplianceTracker, aggregate_force_compliance_v2
from .gates import GENTLE_30N_GATE
from .gym_env import Go2StandBalanceGymEnv
from .rl_env import Go2StandBalanceEnv, StandBalanceEnvConfig


FORCE_LATERAL_DRIFT_MAX_M = GENTLE_30N_GATE.max_force_abs_lateral_displacement
FORCE_LATERAL_DRIFT_PENALTY_SCALE = 160.0
GENTLE_POST_FORCE_FORWARD_MIN_MPS = GENTLE_30N_GATE.min_post_force_forward
GENTLE_POST_FORCE_LATERAL_MAX_MPS = GENTLE_30N_GATE.max_post_force_abs_lateral
GENTLE_POST_FORCE_YAW_MAX_RAD_S = GENTLE_30N_GATE.max_post_force_abs_yaw
GENTLE_SAFETY_COMPLIANCE_MIN = GENTLE_30N_GATE.min_force_step_compliance


def default_vecnormalize_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}_vecnormalize.pkl")


def initialize_training_vecnormalize(
    *,
    vec_env: Any,
    normalize_observation: bool,
    normalize_reward: bool,
    clip_observation: float,
    init_path: str | Path | None,
    freeze: bool,
    prefix_count: float = 1024.0,
    observation_names: Sequence[str] | None = None,
    reset_obs_patterns: Sequence[str] = (),
    reset_obs_std: float = 1.0,
) -> tuple[Any, Path | None, str]:
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    if init_path is None:
        if freeze:
            raise ValueError("--freeze-vecnormalize requires loaded VecNormalize stats")
        return (
            VecNormalize(
                vec_env,
                norm_obs=normalize_observation,
                norm_reward=normalize_reward,
                clip_obs=clip_observation,
            ),
            None,
            "new",
        )

    stats_path = Path(init_path)
    try:
        normalized = VecNormalize.load(str(stats_path), vec_env)
        sync_vecnormalize_wrapper_spaces(normalized, vec_env)
        mode = "exact"
    except Exception as exc:
        normalized = VecNormalize(
            vec_env,
            norm_obs=normalize_observation,
            norm_reward=normalize_reward,
            clip_obs=clip_observation,
        )
        copied_dims = _copy_vecnormalize_prefix_stats(
            normalized,
            stats_path,
            prefix_count=prefix_count,
        )
        if copied_dims <= 0:
            raise ValueError(f"could not initialize VecNormalize from {stats_path}: {exc}") from exc
        if freeze:
            raise ValueError(
                "--freeze-vecnormalize cannot be used with prefix VecNormalize initialization; "
                "new observation dimensions need live statistics"
            )
        mode = f"prefix:{copied_dims}:count={max(float(prefix_count), 1.0):.0f}"

    if reset_obs_patterns:
        if observation_names is None:
            raise ValueError("VecNormalize obs stat reset requires observation names")
        reset_count = reset_vecnormalize_observation_stats(
            normalized,
            observation_names=observation_names,
            name_patterns=reset_obs_patterns,
            std=reset_obs_std,
        )
        mode = f"{mode}+reset:{reset_count}"

    normalized.training = not freeze
    normalized.norm_obs = normalize_observation
    normalized.norm_reward = normalize_reward
    normalized.clip_obs = clip_observation
    return normalized, stats_path, mode


def sync_vecnormalize_wrapper_spaces(normalized: Any, vec_env: Any) -> None:
    if hasattr(vec_env, "observation_space"):
        normalized.observation_space = vec_env.observation_space
    if hasattr(vec_env, "action_space"):
        normalized.action_space = vec_env.action_space


def reset_vecnormalize_observation_stats(
    normalized: Any,
    *,
    observation_names: Sequence[str],
    name_patterns: Sequence[str],
    std: float = 1.0,
) -> int:
    obs_rms = getattr(normalized, "obs_rms", None)
    if obs_rms is None or isinstance(obs_rms, dict):
        return 0

    mean = obs_rms.mean.reshape(-1)
    var = obs_rms.var.reshape(-1)
    if len(observation_names) != mean.shape[0]:
        raise ValueError(
            "observation name count does not match VecNormalize stats: "
            f"{len(observation_names)} != {mean.shape[0]}"
        )

    patterns = tuple(pattern for pattern in name_patterns if pattern)
    if not patterns:
        return 0

    reset_var = max(float(std), 1e-6) ** 2
    reset_count = 0
    for index, name in enumerate(observation_names):
        if any(pattern in name for pattern in patterns):
            mean[index] = 0.0
            var[index] = reset_var
            reset_count += 1

    obs_rms.mean = mean.reshape(obs_rms.mean.shape)
    obs_rms.var = var.reshape(obs_rms.var.shape)
    return reset_count


def _copy_vecnormalize_prefix_stats(target: Any, source_path: Path, *, prefix_count: float) -> int:
    with source_path.open("rb") as file_handle:
        source = pickle.load(file_handle)

    source_obs_rms = getattr(source, "obs_rms", None)
    target_obs_rms = getattr(target, "obs_rms", None)
    if source_obs_rms is None or target_obs_rms is None:
        return 0
    if isinstance(source_obs_rms, dict) or isinstance(target_obs_rms, dict):
        return 0

    source_mean = source_obs_rms.mean.reshape(-1)
    source_var = source_obs_rms.var.reshape(-1)
    target_mean = target_obs_rms.mean.reshape(-1)
    target_var = target_obs_rms.var.reshape(-1)
    copied_dims = min(source_mean.shape[0], target_mean.shape[0])
    if copied_dims <= 0:
        return 0

    target_mean[:copied_dims] = source_mean[:copied_dims]
    target_var[:copied_dims] = source_var[:copied_dims]
    target_obs_rms.mean = target_mean.reshape(target_obs_rms.mean.shape)
    target_obs_rms.var = target_var.reshape(target_obs_rms.var.shape)
    target_obs_rms.count = max(float(prefix_count), 1.0)

    source_ret_rms = getattr(source, "ret_rms", None)
    target_ret_rms = getattr(target, "ret_rms", None)
    if source_ret_rms is not None and target_ret_rms is not None:
        target_ret_rms.mean = source_ret_rms.mean.copy()
        target_ret_rms.var = source_ret_rms.var.copy()
        target_ret_rms.count = float(source_ret_rms.count)

    return copied_dims


def make_vec_env(
    *,
    model_path: str | Path,
    config: StandBalanceEnvConfig,
    num_envs: int = 1,
) -> Any:
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    num_envs = max(1, int(num_envs))

    def _factory() -> Monitor:
        return Monitor(Go2StandBalanceGymEnv(model_path=model_path, config=config))

    return DummyVecEnv([_factory for _ in range(num_envs)])


def load_vecnormalize_if_available(
    *,
    vec_env: Any,
    checkpoint_path: str | Path,
    vecnormalize_path: str | Path | None = None,
    disabled: bool = False,
) -> tuple[Any, Path | None]:
    if disabled:
        return vec_env, None

    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    explicit_path = vecnormalize_path is not None
    stats_path = Path(vecnormalize_path) if explicit_path else default_vecnormalize_path(checkpoint_path)
    if not stats_path.exists():
        if explicit_path:
            raise FileNotFoundError(f"VecNormalize stats not found: {stats_path}")
        return vec_env, None

    normalized = VecNormalize.load(str(stats_path), vec_env)
    sync_vecnormalize_wrapper_spaces(normalized, vec_env)
    normalized.training = False
    normalized.norm_reward = False
    return normalized, stats_path


def unwrap_go2_env(vec_env: Any) -> Go2StandBalanceEnv:
    current = getattr(vec_env, "venv", vec_env)
    if hasattr(current, "envs"):
        current = current.envs[0]

    while current is not None:
        if isinstance(current, Go2StandBalanceGymEnv):
            return current.env
        if isinstance(current, Go2StandBalanceEnv):
            return current
        current = getattr(current, "env", None)

    raise RuntimeError("could not unwrap Go2StandBalanceEnv from vectorized environment")


class RewardTermLoggingCallback:
    """Record per-step reward components from env infos into the SB3 logger."""

    def __init__(self, *, window_size: int = 4096, log_interval: int = 512) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError(
                "Stable-Baselines3 is missing. Install training dependencies with: "
                "pip install -e '.[train]'"
            ) from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: RewardTermLoggingCallback) -> None:
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer._on_step(self)

        self._callback = _Callback(self)
        self.window_size = window_size
        self.log_interval = max(1, log_interval)
        self._buffers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window_size))

    @property
    def callback(self) -> Any:
        return self._callback

    def _on_step(self, callback: Any) -> bool:
        infos = callback.locals.get("infos", [])
        for info in infos:
            for name, value in (info.get("reward_terms") or {}).items():
                self._buffers[f"reward/{name}"].append(float(value))
            for name, value in (info.get("reward_raw_terms") or {}).items():
                self._buffers[f"reward_raw/{name}"].append(float(value))

        if callback.n_calls % self.log_interval == 0:
            for key, values in sorted(self._buffers.items()):
                if values:
                    callback.logger.record(key, sum(values) / len(values))
        return True


class ExternalForceCurriculumCallback:
    """Ramp random external force magnitude through the wrapped environments."""

    def __init__(self, *, total_steps: int) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError(
                "Stable-Baselines3 is missing. Install training dependencies with: "
                "pip install -e '.[train]'"
            ) from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: ExternalForceCurriculumCallback) -> None:
                super().__init__()
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer._set_progress(self, 0.0)

            def _on_step(self) -> bool:
                return self.outer._on_step(self)

        self._callback = _Callback(self)
        self.total_steps = max(1, int(total_steps))

    @property
    def callback(self) -> Any:
        return self._callback

    def _on_step(self, callback: Any) -> bool:
        progress = min(float(callback.num_timesteps) / float(self.total_steps), 1.0)
        self._set_progress(callback, progress)
        callback.logger.record("external_force/curriculum_progress", progress)
        return True

    def _set_progress(self, callback: Any, progress: float) -> None:
        callback.training_env.env_method("set_external_force_curriculum_progress", progress)


class LocomotionEvalCallback:
    """Periodically evaluate locomotion metrics and save the best checkpoint."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        config: StandBalanceEnvConfig,
        eval_freq: int,
        best_model_path: str | Path,
        report_path: str | Path | None = None,
        episodes: int = 3,
        duration: float = 8.0,
        deterministic: bool = True,
        seed: int = 1000,
        eval_suite: str = "single",
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError(
                "Stable-Baselines3 is missing. Install training dependencies with: "
                "pip install -e '.[train]'"
            ) from exc

        class _Callback(BaseCallback):
            def __init__(self, outer: LocomotionEvalCallback) -> None:
                super().__init__()
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer._on_training_start(self)

            def _on_step(self) -> bool:
                return self.outer._on_step(self)

        self._callback = _Callback(self)
        self.model_path = Path(model_path)
        self.config = config
        self.eval_freq = max(1, int(eval_freq))
        self.best_model_path = Path(best_model_path)
        self.report_path = Path(report_path) if report_path else self.best_model_path.with_suffix(".eval.json")
        self.latest_report_path = self.report_path.with_name(
            f"{self.report_path.stem}.latest{self.report_path.suffix}"
        )
        self.episodes = max(1, int(episodes))
        self.duration = float(duration)
        self.deterministic = deterministic
        self.seed = int(seed)
        self.eval_suite = eval_suite
        self.best_score = float("-inf")
        self.last_eval_timestep = 0

    @property
    def callback(self) -> Any:
        return self._callback

    def _on_training_start(self, callback: Any) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        for output_format in callback.logger.output_formats:
            if isinstance(output_format, HumanOutputFormat):
                output_format.max_length = max(output_format.max_length, 96)

    def _on_step(self, callback: Any) -> bool:
        if callback.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True

        self.last_eval_timestep = int(callback.num_timesteps)
        if self.eval_suite == "gentle_30n":
            scenario_reports = {}
            for index, (name, config) in enumerate(build_gentle_30n_eval_configs(self.config).items()):
                scenario_reports[name] = evaluate_locomotion_policy(
                    model=callback.model,
                    normalizer=callback.training_env,
                    model_path=self.model_path,
                    config=config,
                    episodes=self.episodes,
                    duration=self.duration,
                    deterministic=self.deterministic,
                    seed=self.seed + index * 1000,
                )
            report = combine_locomotion_eval_reports(scenario_reports)
            for scenario_name, scenario_report in scenario_reports.items():
                for name, value in scenario_report["aggregate"].items():
                    if isinstance(value, (float, int)):
                        callback.logger.record(
                            f"eval/{scenario_name}/{name}",
                            float(value),
                        )
        else:
            report = evaluate_locomotion_policy(
                model=callback.model,
                normalizer=callback.training_env,
                model_path=self.model_path,
                config=self.config,
                episodes=self.episodes,
                duration=self.duration,
                deterministic=self.deterministic,
                seed=self.seed + self.last_eval_timestep,
            )
        aggregate = report["aggregate"]
        for name, value in aggregate.items():
            if isinstance(value, (float, int)):
                prefix = "eval/combined" if self.eval_suite == "gentle_30n" else "eval"
                callback.logger.record(f"{prefix}/{name}", float(value))

        score = float(aggregate["score"])
        success_rate = float(aggregate.get("success_rate", 0.0) or 0.0)
        best_eligible = float(aggregate.get("best_eligible", 1.0) or 0.0)
        self.latest_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if best_eligible > 0.0 and success_rate > 0.0 and score > self.best_score:
            self.best_score = score
            self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
            callback.model.save(str(self.best_model_path))
            if hasattr(callback.training_env, "save"):
                callback.training_env.save(str(default_vecnormalize_path(self.best_model_path)))
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if self.eval_suite == "gentle_30n":
                print(
                    "saved best gentle_30n eval checkpoint "
                    f"{self.best_model_path} score={score:.3f}"
                )
            else:
                print(
                    "saved best eval checkpoint "
                    f"{self.best_model_path} score={score:.3f} "
                    f"vx={aggregate['mean_forward_velocity']:.3f} "
                    f"lateral={aggregate['mean_abs_lateral_velocity']:.3f} "
                    f"yaw_error={aggregate['mean_yaw_rate_error']:.3f}"
                )
        return True


def build_gentle_30n_eval_configs(
    config: StandBalanceEnvConfig,
) -> dict[str, StandBalanceEnvConfig]:
    common = replace(
        config,
        randomize_commands=False,
        external_force_active_body_count=1,
        external_force_min_n=30.0,
        external_force_max_n=30.0,
        external_force_curriculum_start_n=None,
        external_force_torque_max_nm=0.0,
        external_force_direction_mode="default",
        external_force_net_force_limit_n=30.0,
        external_force_net_torque_limit_nm=20.0,
    )
    return {
        "no_force": replace(
            common,
            external_force_probability=0.0,
            external_force_min_n=0.0,
            external_force_max_n=0.0,
            external_force_direction_angle_rad=None,
            external_force_z_fraction=0.0,
        ),
        "lateral_plus_90": replace(
            common,
            external_force_probability=1.0,
            external_force_direction_angle_rad=math.pi / 2.0,
            external_force_z_fraction=0.0,
        ),
        "lateral_minus_90": replace(
            common,
            external_force_probability=1.0,
            external_force_direction_angle_rad=-math.pi / 2.0,
            external_force_z_fraction=0.0,
        ),
        "random_horizontal": replace(
            common,
            external_force_probability=1.0,
            external_force_direction_angle_rad=None,
            external_force_z_fraction=0.0,
        ),
        "random_3d": replace(
            common,
            external_force_probability=1.0,
            external_force_direction_angle_rad=None,
            external_force_z_fraction=max(float(config.external_force_z_fraction), 0.08),
        ),
    }


def combine_locomotion_eval_reports(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    scores: list[float] = []
    success_rates: list[float] = []

    for scenario_name, report in reports.items():
        aggregate = report["aggregate"]
        scores.append(float(aggregate.get("score", 0.0) or 0.0))
        success_rates.append(float(aggregate.get("success_rate", 0.0) or 0.0))
        if scenario_name == "no_force":
            if float(aggregate.get("success_rate", 0.0) or 0.0) < 1.0:
                failures.append(f"{scenario_name}.success_rate")
            if float(aggregate.get("best_eligible", 0.0) or 0.0) < 1.0:
                failures.append(f"{scenario_name}.best_eligible")
            continue

        gates = (
            ("success_rate", float(aggregate.get("success_rate", 0.0) or 0.0) >= 0.90),
            ("fall_rate", float(aggregate.get("fall_rate", 1.0) or 0.0) <= 0.10),
            (
                "external_force_recovery_success_rate",
                float(aggregate.get("external_force_recovery_success_rate", 0.0) or 0.0) >= 0.90,
            ),
            (
                "post_force_mean_forward_velocity",
                float(aggregate.get("post_force_mean_forward_velocity", 0.0) or 0.0)
                >= GENTLE_POST_FORCE_FORWARD_MIN_MPS,
            ),
            (
                "post_force_mean_abs_lateral_velocity",
                float(aggregate.get("post_force_mean_abs_lateral_velocity", float("inf")))
                <= GENTLE_POST_FORCE_LATERAL_MAX_MPS,
            ),
            (
                "post_force_mean_abs_yaw_rate",
                float(aggregate.get("post_force_mean_abs_yaw_rate", float("inf")))
                <= GENTLE_POST_FORCE_YAW_MAX_RAD_S,
            ),
            (
                "mean_abs_final_lateral_displacement",
                float(aggregate.get("mean_abs_final_lateral_displacement", float("inf")))
                <= FORCE_LATERAL_DRIFT_MAX_M,
            ),
            (
                "force_active_step_compliance_rate",
                float(aggregate.get("force_active_step_compliance_rate", 0.0) or 0.0)
                >= GENTLE_SAFETY_COMPLIANCE_MIN,
            ),
            (
                "episode_compliance_rate",
                float(aggregate.get("episode_compliance_rate", 0.0) or 0.0)
                >= GENTLE_SAFETY_COMPLIANCE_MIN,
            ),
        )
        failures.extend(
            f"{scenario_name}.{metric}"
            for metric, passed in gates
            if not passed
        )

    combined_score = _mean(scores)
    return {
        "suite": "gentle_30n",
        "scenarios": reports,
        "failures": failures,
        "aggregate": {
            "scenario_count": len(reports),
            "score": combined_score,
            "worst_scenario_score": min(scores) if scores else 0.0,
            "success_rate": min(success_rates) if success_rates else 0.0,
            "best_eligible": 0.0 if failures else 1.0,
        },
    }


def evaluate_locomotion_policy(
    *,
    model: Any,
    normalizer: Any | None,
    model_path: str | Path,
    config: StandBalanceEnvConfig,
    episodes: int,
    duration: float,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    eval_config = replace(config, episode_length_s=float(duration))
    env = Go2StandBalanceGymEnv(model_path=model_path, config=eval_config)
    episode_reports = [
        _evaluate_locomotion_episode(
            env=env,
            model=model,
            normalizer=normalizer,
            deterministic=deterministic,
            seed=seed + index,
        )
        for index in range(max(1, episodes))
    ]
    return {
        "task": eval_config.task,
        "model": str(model_path),
        "config": {
            "duration": duration,
            "control_dt": eval_config.control_dt,
            "target_forward_velocity": eval_config.target_forward_velocity,
            "target_lateral_velocity": eval_config.target_lateral_velocity,
            "target_yaw_rate": eval_config.target_yaw_rate,
            "action_scale": eval_config.action_scale,
            "action_smoothing": eval_config.action_smoothing,
            "reward_scale": eval_config.reward_scale,
            "velocity_pose_profile": eval_config.velocity_pose_profile,
            "velocity_reward_profile": eval_config.velocity_reward_profile,
            "velocity_command_frame": eval_config.velocity_command_frame,
            "randomize_commands": eval_config.randomize_commands,
            "external_force_mode": eval_config.external_force_mode,
            "external_force_probability": eval_config.external_force_probability,
            "external_force_body_names": eval_config.external_force_body_names,
            "external_force_active_body_count": eval_config.external_force_active_body_count,
            "external_force_start_s_range": eval_config.external_force_start_s_range,
            "external_force_duration_s_range": eval_config.external_force_duration_s_range,
            "external_force_min_n": eval_config.external_force_min_n,
            "external_force_max_n": eval_config.external_force_max_n,
            "external_force_z_fraction": eval_config.external_force_z_fraction,
            "external_force_direction_angle_rad": eval_config.external_force_direction_angle_rad,
            "external_force_direction_mode": eval_config.external_force_direction_mode,
            "external_force_lateral_probability": eval_config.external_force_lateral_probability,
            "external_force_torque_max_nm": eval_config.external_force_torque_max_nm,
            "external_force_spring_stiffness_range": eval_config.external_force_spring_stiffness_range,
            "external_force_spring_damping": eval_config.external_force_spring_damping,
            "external_force_guiding_probability": eval_config.external_force_guiding_probability,
            "external_force_transition_s": eval_config.external_force_transition_s,
            "external_force_net_force_limit_n": eval_config.external_force_net_force_limit_n,
            "external_force_net_torque_limit_nm": eval_config.external_force_net_torque_limit_nm,
            "external_force_reference_mass": eval_config.external_force_reference_mass,
            "external_force_reference_damping": eval_config.external_force_reference_damping,
            "external_force_reference_velocity_clip": eval_config.external_force_reference_velocity_clip,
            "external_force_reference_acceleration_clip": eval_config.external_force_reference_acceleration_clip,
            "external_force_safe_limit_min_n": eval_config.external_force_safe_limit_min_n,
            "external_force_safe_limit_max_n": eval_config.external_force_safe_limit_max_n,
            "external_force_safe_margin_n": eval_config.external_force_safe_margin_n,
            "include_external_force_observation": eval_config.include_external_force_observation,
        },
        "aggregate": _aggregate_locomotion_episodes(episode_reports, eval_config),
        "episodes": episode_reports,
    }


def _evaluate_locomotion_episode(
    *,
    env: Go2StandBalanceGymEnv,
    model: Any,
    normalizer: Any | None,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    steps = 0
    terminated = False
    truncated = False
    total_reward = 0.0
    base_z_values: list[float] = []
    forward_velocities: list[float] = []
    command_speeds: list[float] = []
    along_command_velocities: list[float] = []
    lateral_velocities: list[float] = []
    yaw_rates: list[float] = []
    base_yaws: list[float] = []
    base_positions: list[tuple[float, float, float]] = []
    yaw_rate_errors: list[float] = []
    orientation_errors: list[float] = []
    onnx_action_norms: list[float] = []
    residual_action_norms: list[float] = []
    final_action_norms: list[float] = []
    final_minus_onnx_action_norms: list[float] = []
    force_active_residual_action_norms: list[float] = []
    force_active_final_minus_onnx_action_norms: list[float] = []
    force_onset_residual_action_norms: list[float] = []
    force_onset_external_force_excess_values: list[float] = []
    action_balance_values = defaultdict(list)
    external_force_steps = 0
    external_force_magnitudes: list[float] = []
    external_force_torque_magnitudes: list[float] = []
    external_force_excess_values: list[float] = []
    external_force_step_compliance: list[bool] = []
    external_force_safe_limit_n = 0.0
    external_force_seen = False
    post_force_forward_velocities: list[float] = []
    post_force_abs_lateral_velocities: list[float] = []
    post_force_abs_yaw_rates: list[float] = []
    contact_counts = defaultdict(int)
    last_info: dict[str, Any] = {}
    force_onset_window_s = max(
        0.0,
        float(env.env.config.velocity_reward.external_force_excess_onset_window_s),
        float(env.env.config.velocity_reward.unsafe_force_onset_window_s),
    )
    force_compliance = ForceComplianceTracker(
        dt_s=env.env.config.control_dt,
        onset_window_s=force_onset_window_s,
    )

    while not terminated and not truncated:
        policy_obs = normalizer.normalize_obs(obs) if hasattr(normalizer, "normalize_obs") else obs
        action, _ = model.predict(policy_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        force_compliance.observe(info)
        total_reward += float(reward)
        steps += 1

        base_z_values.append(float(info.get("base_z", 0.0)))
        base_linear_velocity_world = info.get("base_linear_velocity") or [0.0, 0.0, 0.0]
        base_linear_velocity = info.get("base_linear_velocity_tracking_frame") or base_linear_velocity_world
        base_angular_velocity = info.get("base_angular_velocity") or [0.0, 0.0, 0.0]
        command = info.get("command") or [0.0, 0.0, 0.0]
        projected_gravity = info.get("projected_gravity") or [0.0, 0.0, -1.0]
        last_action = info.get("last_action") or []
        action_names = info.get("action_names") or []
        onnx_action = info.get("onnx_action")
        residual_action = info.get("residual_action")
        final_action = info.get("final_action")
        onnx_action_norm = _vector_norm(onnx_action)
        residual_action_norm = _vector_norm(residual_action)
        final_action_norm = _vector_norm(final_action)
        final_minus_onnx_action_norm = _vector_delta_norm(final_action, onnx_action)
        if onnx_action_norm is not None:
            onnx_action_norms.append(onnx_action_norm)
        if residual_action_norm is not None:
            residual_action_norms.append(residual_action_norm)
        if final_action_norm is not None:
            final_action_norms.append(final_action_norm)
        if final_minus_onnx_action_norm is not None:
            final_minus_onnx_action_norms.append(final_minus_onnx_action_norm)
        base_yaw = info.get("base_yaw")
        base_position = info.get("base_position") or []
        if len(base_position) >= 3:
            base_positions.append(
                (float(base_position[0]), float(base_position[1]), float(base_position[2]))
            )
        forward_velocities.append(float(base_linear_velocity[0]))
        lateral_velocities.append(float(base_linear_velocity[1]))
        yaw_rates.append(float(base_angular_velocity[2]))
        if base_yaw is not None:
            base_yaws.append(float(base_yaw))
        command_x = float(command[0])
        command_y = float(command[1])
        command_speed = math.sqrt(command_x * command_x + command_y * command_y)
        command_speeds.append(command_speed)
        if command_speed > 1e-6:
            along_command_velocities.append(
                (float(base_linear_velocity[0]) * command_x + float(base_linear_velocity[1]) * command_y)
                / command_speed
            )
        else:
            along_command_velocities.append(0.0)
        yaw_rate_errors.append(abs(float(command[2]) - float(base_angular_velocity[2])))
        orientation_errors.append(
            math.sqrt(float(projected_gravity[0]) ** 2 + float(projected_gravity[1]) ** 2)
        )
        if last_action:
            for name, value in action_balance_metrics(last_action, action_names).items():
                action_balance_values[name].append(value)
        if bool(info.get("external_force_active", False)):
            external_force_seen = True
            external_force_steps += 1
            external_force_safe_limit_n = float(info.get("external_force_safe_limit_n", 0.0))
            external_force_magnitudes.append(float(info.get("external_force_magnitude", 0.0)))
            external_force_torque_magnitudes.append(float(info.get("external_force_torque_magnitude", 0.0)))
            external_force_excess_values.append(float(info.get("external_force_excess_n", 0.0)))
            if residual_action_norm is not None:
                force_active_residual_action_norms.append(residual_action_norm)
            if final_minus_onnx_action_norm is not None:
                force_active_final_minus_onnx_action_norms.append(final_minus_onnx_action_norm)
            force_start_s = info.get("external_force_start_s")
            force_elapsed_s = (
                float(info.get("t", 0.0)) - float(force_start_s)
                if force_start_s is not None
                else None
            )
            if force_elapsed_s is not None and 0.0 <= force_elapsed_s <= force_onset_window_s:
                if residual_action_norm is not None:
                    force_onset_residual_action_norms.append(residual_action_norm)
                force_onset_external_force_excess_values.append(
                    float(info.get("external_force_excess_n", 0.0))
                )
            compliant = info.get("external_force_step_compliant")
            if compliant is not None:
                external_force_step_compliance.append(bool(compliant))
        elif external_force_seen:
            post_force_forward_velocities.append(float(base_linear_velocity[0]))
            post_force_abs_lateral_velocities.append(abs(float(base_linear_velocity[1])))
            post_force_abs_yaw_rates.append(abs(float(base_angular_velocity[2])))
        for foot, active in (info.get("contacts") or {}).items():
            if active:
                contact_counts[str(foot)] += 1

    divisor = max(steps, 1)
    fall_base_z = env.env.config.fall_base_z
    fall_detected = bool(base_z_values and min(base_z_values) < fall_base_z)
    failure_terminated = bool(terminated and not truncated)
    contact_ratio = {
        foot: contact_counts[foot] / divisor
        for foot in ("FR", "FL", "RR", "RL")
    }
    report = {
        "seed": seed,
        "steps": steps,
        "duration": steps * env.env.config.control_dt,
        "total_reward": total_reward,
        "fall_detected": fall_detected or failure_terminated,
        "external_force_applied": external_force_steps > 0,
        "external_force_steps": external_force_steps,
        "external_force_max_magnitude": max(external_force_magnitudes) if external_force_magnitudes else 0.0,
        "external_force_max_torque_magnitude": (
            max(external_force_torque_magnitudes) if external_force_torque_magnitudes else 0.0
        ),
        "external_force_safe_limit_n": external_force_safe_limit_n,
        "external_force_mean_excess_n": _mean(external_force_excess_values),
        "external_force_max_excess_n": (
            max(external_force_excess_values) if external_force_excess_values else 0.0
        ),
        "force_active_step_compliance_rate": (
            sum(external_force_step_compliance) / len(external_force_step_compliance)
            if external_force_step_compliance
            else 1.0
        ),
        "episode_compliant": (
            sum(external_force_step_compliance) / len(external_force_step_compliance) >= 0.95
            if external_force_step_compliance
            else True
        ),
        "post_force_mean_forward_velocity": _mean(post_force_forward_velocities),
        "post_force_mean_abs_lateral_velocity": _mean(post_force_abs_lateral_velocities),
        "post_force_mean_abs_yaw_rate": _mean(post_force_abs_yaw_rates),
        "command": last_info.get("command") or [0.0, 0.0, 0.0],
        "base_z_min": min(base_z_values) if base_z_values else None,
        "mean_forward_velocity": _mean(forward_velocities),
        "mean_command_speed": _mean(command_speeds),
        "mean_along_command_velocity": _mean(along_command_velocities),
        "mean_lateral_velocity": _mean(lateral_velocities),
        "mean_abs_lateral_velocity": _mean([abs(value) for value in lateral_velocities]),
        "mean_yaw_rate": _mean(yaw_rates),
        "mean_abs_yaw_rate": _mean([abs(value) for value in yaw_rates]),
        "final_base_yaw": base_yaws[-1] if base_yaws else None,
        "final_lateral_displacement": (
            base_positions[-1][1] - base_positions[0][1]
            if len(base_positions) >= 2
            else None
        ),
        "abs_final_lateral_displacement": (
            abs(base_positions[-1][1] - base_positions[0][1])
            if len(base_positions) >= 2
            else None
        ),
        "mean_yaw_rate_error": _mean(yaw_rate_errors),
        "mean_orientation_error": _mean(orientation_errors),
        "mean_onnx_action_norm": _mean(onnx_action_norms),
        "mean_residual_action_norm": _mean(residual_action_norms),
        "mean_final_action_norm": _mean(final_action_norms),
        "mean_final_minus_onnx_action_norm": _mean(final_minus_onnx_action_norms),
        "force_active_mean_residual_action_norm": _mean_or_none(force_active_residual_action_norms),
        "force_active_max_residual_action_norm": _max_or_none(force_active_residual_action_norms),
        "force_active_mean_final_minus_onnx_action_norm": _mean_or_none(
            force_active_final_minus_onnx_action_norms
        ),
        "force_onset_mean_residual_action_norm": _mean_or_none(force_onset_residual_action_norms),
        "force_onset_max_external_force_excess_n": _max_or_none(force_onset_external_force_excess_values),
        "contact_ratio": contact_ratio,
        "contact_imbalance": _contact_imbalance(contact_ratio),
    }
    report.update(force_compliance.summary())
    report.update(contact_balance_metrics(contact_ratio))
    report.update({
        name: _mean(action_balance_values[name])
        for name in ACTION_BALANCE_KEYS
    })
    return report


def _aggregate_locomotion_episodes(
    episodes: list[dict[str, Any]],
    config: StandBalanceEnvConfig,
) -> dict[str, float | int | None]:
    count = max(len(episodes), 1)
    fall_rate = sum(bool(episode["fall_detected"]) for episode in episodes) / count
    success_rate = 1.0 - fall_rate
    mean_forward_velocity = _mean([episode["mean_forward_velocity"] for episode in episodes])
    mean_command_speed = _mean([episode["mean_command_speed"] for episode in episodes])
    mean_along_command_velocity = _mean([episode["mean_along_command_velocity"] for episode in episodes])
    mean_lateral_velocity = _mean([episode["mean_lateral_velocity"] for episode in episodes])
    mean_abs_lateral_velocity = _mean([episode["mean_abs_lateral_velocity"] for episode in episodes])
    mean_yaw_rate = _mean([episode["mean_yaw_rate"] for episode in episodes])
    mean_abs_yaw_rate = _mean([episode.get("mean_abs_yaw_rate") for episode in episodes])
    mean_final_base_yaw = _mean([episode["final_base_yaw"] for episode in episodes])
    mean_abs_final_lateral_displacement = _mean([
        episode.get("abs_final_lateral_displacement")
        for episode in episodes
    ])
    mean_yaw_rate_error = _mean([episode["mean_yaw_rate_error"] for episode in episodes])
    mean_orientation_error = _mean([episode["mean_orientation_error"] for episode in episodes])
    mean_onnx_action_norm = _mean([
        episode.get("mean_onnx_action_norm")
        for episode in episodes
    ])
    mean_residual_action_norm = _mean([
        episode.get("mean_residual_action_norm")
        for episode in episodes
    ])
    mean_final_action_norm = _mean([
        episode.get("mean_final_action_norm")
        for episode in episodes
    ])
    mean_final_minus_onnx_action_norm = _mean([
        episode.get("mean_final_minus_onnx_action_norm")
        for episode in episodes
    ])
    mean_force_active_residual_action_norm = _mean_or_none([
        episode.get("force_active_mean_residual_action_norm")
        for episode in episodes
    ])
    max_force_active_residual_action_norm = _max_or_none([
        episode.get("force_active_max_residual_action_norm")
        for episode in episodes
    ])
    mean_force_active_final_minus_onnx_action_norm = _mean_or_none([
        episode.get("force_active_mean_final_minus_onnx_action_norm")
        for episode in episodes
    ])
    mean_force_onset_residual_action_norm = _mean_or_none([
        episode.get("force_onset_mean_residual_action_norm")
        for episode in episodes
    ])
    max_force_onset_external_force_excess_n = _max_or_none([
        episode.get("force_onset_max_external_force_excess_n")
        for episode in episodes
    ])
    mean_contact_imbalance = _mean([episode["contact_imbalance"] for episode in episodes])
    external_force_recovery_success_rate = sum(
        bool(episode["external_force_applied"]) and not bool(episode["fall_detected"])
        for episode in episodes
    ) / count
    mean_external_force_max_magnitude = _mean([
        episode["external_force_max_magnitude"]
        for episode in episodes
    ])
    mean_external_force_max_torque_magnitude = _mean([
        episode["external_force_max_torque_magnitude"]
        for episode in episodes
    ])
    mean_external_force_safe_limit_n = _mean([
        episode.get("external_force_safe_limit_n")
        for episode in episodes
    ])
    mean_external_force_excess_n = _mean([
        episode.get("external_force_mean_excess_n")
        for episode in episodes
    ])
    max_external_force_excess_n = max(
        [float(episode.get("external_force_max_excess_n", 0.0)) for episode in episodes]
        or [0.0]
    )
    force_active_step_compliance_rate = _mean([
        episode.get("force_active_step_compliance_rate")
        for episode in episodes
    ])
    episode_compliance_rate = sum(
        bool(episode.get("episode_compliant", True))
        for episode in episodes
    ) / count
    post_force_mean_forward_velocity = _mean([
        episode.get("post_force_mean_forward_velocity")
        for episode in episodes
    ])
    post_force_mean_abs_lateral_velocity = _mean([
        episode.get("post_force_mean_abs_lateral_velocity")
        for episode in episodes
    ])
    post_force_mean_abs_yaw_rate = _mean([
        episode.get("post_force_mean_abs_yaw_rate")
        for episode in episodes
    ])
    mean_command_forward_velocity = _mean([
        abs(float((episode.get("command") or [config.target_forward_velocity])[0]))
        for episode in episodes
    ])
    target_speed = max(abs(float(config.target_forward_velocity)), mean_command_forward_velocity)
    target_speed = max(target_speed, mean_command_speed)
    speed_scale = max(target_speed, 0.05)
    speed_tracking = max(0.0, 1.0 - abs(target_speed - mean_along_command_velocity) / speed_scale)
    forward_progress = max(0.0, min(mean_along_command_velocity / speed_scale, 1.0))
    locomotion_score = 0.5 * speed_tracking + 0.5 * forward_progress
    no_force_gate_penalty = 0.0
    force_eval = config.external_force_probability > 0.0 and config.external_force_max_n > 0.0
    safe_limit_eval = force_eval and config.external_force_safe_limit_max_n > 0.0
    force_lateral_drift_success_rate = success_rate
    force_lateral_drift_penalty = 0.0
    straight_no_force_eval = (
        not config.randomize_commands
        and config.external_force_probability <= 0.0
        and target_speed > 0.05
        and abs(float(config.target_lateral_velocity)) <= 0.05
        and abs(float(config.target_yaw_rate)) <= 0.05
    )
    if straight_no_force_eval:
        straightness_success_rate = sum(
            (
                not bool(episode["fall_detected"])
                and float(episode["mean_forward_velocity"]) >= 0.35
                and float(episode["mean_forward_velocity"]) <= target_speed + 0.05
                and float(episode["mean_abs_lateral_velocity"]) <= 0.08
                and float(episode.get("mean_abs_yaw_rate") or episode["mean_yaw_rate_error"]) <= 0.12
                and episode.get("final_base_yaw") is not None
                and abs(float(episode["final_base_yaw"])) <= 0.35
                and episode.get("abs_final_lateral_displacement") is not None
                and float(episode["abs_final_lateral_displacement"]) <= 0.35
            )
            for episode in episodes
        ) / count
        no_force_gate_penalty = (
            250.0 * max(0.0, mean_abs_lateral_velocity - 0.08)
            + 400.0 * max(0.0, mean_abs_yaw_rate - 0.12)
            + 250.0 * max(0.0, mean_abs_final_lateral_displacement - 0.35)
            + 80.0 * max(0.0, abs(target_speed - mean_along_command_velocity) - 0.05)
            + 160.0 * max(0.0, mean_forward_velocity - target_speed - 0.05)
        )
    else:
        straightness_success_rate = success_rate
    if force_eval:
        force_lateral_drift_success_rate = sum(
            (
                not bool(episode["fall_detected"])
                and episode.get("abs_final_lateral_displacement") is not None
                and float(episode["abs_final_lateral_displacement"]) <= FORCE_LATERAL_DRIFT_MAX_M
            )
            for episode in episodes
        ) / count
        force_lateral_drift_penalty = FORCE_LATERAL_DRIFT_PENALTY_SCALE * max(
            0.0,
            mean_abs_final_lateral_displacement - FORCE_LATERAL_DRIFT_MAX_M,
        )
    safety_compliance_penalty = 0.0
    if safe_limit_eval:
        safety_compliance_penalty = (
            100.0 * max(0.0, GENTLE_30N_GATE.min_force_step_compliance - force_active_step_compliance_rate)
            + 100.0 * max(0.0, GENTLE_30N_GATE.min_episode_compliance - episode_compliance_rate)
        )
    score = (
        100.0 * success_rate * locomotion_score
        - 20.0 * mean_abs_lateral_velocity
        - 10.0 * mean_yaw_rate_error
        - 10.0 * mean_orientation_error
        - 5.0 * mean_contact_imbalance
        - no_force_gate_penalty
        - force_lateral_drift_penalty
        - safety_compliance_penalty
    )
    if force_eval:
        score += 30.0 * external_force_recovery_success_rate
    best_eligible = straightness_success_rate >= 0.9 and (
        not force_eval or force_lateral_drift_success_rate >= 0.9
    ) and (
        not safe_limit_eval
        or (
            force_active_step_compliance_rate >= GENTLE_30N_GATE.min_force_step_compliance
            and episode_compliance_rate >= GENTLE_30N_GATE.min_episode_compliance
        )
    )
    aggregate = {
        "episodes": len(episodes),
        "score": score,
        "success_rate": success_rate,
        "fall_rate": fall_rate,
        "mean_steps": _mean([episode["steps"] for episode in episodes]),
        "mean_total_reward": _mean([episode["total_reward"] for episode in episodes]),
        "base_z_min": min(
            episode["base_z_min"]
            for episode in episodes
            if episode["base_z_min"] is not None
        )
        if any(episode["base_z_min"] is not None for episode in episodes)
        else None,
        "mean_forward_velocity": mean_forward_velocity,
        "mean_command_speed": mean_command_speed,
        "mean_along_command_velocity": mean_along_command_velocity,
        "mean_lateral_velocity": mean_lateral_velocity,
        "mean_abs_lateral_velocity": mean_abs_lateral_velocity,
        "mean_yaw_rate": mean_yaw_rate,
        "mean_abs_yaw_rate": mean_abs_yaw_rate,
        "mean_final_base_yaw": mean_final_base_yaw,
        "mean_abs_final_lateral_displacement": mean_abs_final_lateral_displacement,
        "mean_yaw_rate_error": mean_yaw_rate_error,
        "mean_orientation_error": mean_orientation_error,
        "mean_onnx_action_norm": mean_onnx_action_norm,
        "mean_residual_action_norm": mean_residual_action_norm,
        "mean_final_action_norm": mean_final_action_norm,
        "mean_final_minus_onnx_action_norm": mean_final_minus_onnx_action_norm,
        "mean_force_active_residual_action_norm": mean_force_active_residual_action_norm,
        "max_force_active_residual_action_norm": max_force_active_residual_action_norm,
        "mean_force_active_final_minus_onnx_action_norm": mean_force_active_final_minus_onnx_action_norm,
        "mean_force_onset_residual_action_norm": mean_force_onset_residual_action_norm,
        "max_force_onset_external_force_excess_n": max_force_onset_external_force_excess_n,
        "mean_contact_imbalance": mean_contact_imbalance,
        "external_force_recovery_success_rate": external_force_recovery_success_rate,
        "mean_external_force_max_magnitude": mean_external_force_max_magnitude,
        "mean_external_force_max_torque_magnitude": mean_external_force_max_torque_magnitude,
        "mean_external_force_safe_limit_n": mean_external_force_safe_limit_n,
        "mean_external_force_excess_n": mean_external_force_excess_n,
        "max_external_force_excess_n": max_external_force_excess_n,
        "force_active_step_compliance_rate": force_active_step_compliance_rate,
        "episode_compliance_rate": episode_compliance_rate,
        "post_force_mean_forward_velocity": post_force_mean_forward_velocity,
        "post_force_mean_abs_lateral_velocity": post_force_mean_abs_lateral_velocity,
        "post_force_mean_abs_yaw_rate": post_force_mean_abs_yaw_rate,
        "safety_compliance_penalty": safety_compliance_penalty,
        "mean_command_forward_velocity": mean_command_forward_velocity,
        "speed_tracking_score": speed_tracking,
        "forward_progress_score": forward_progress,
        "locomotion_score": locomotion_score,
        "no_force_gate_penalty": no_force_gate_penalty,
        "force_lateral_drift_threshold": FORCE_LATERAL_DRIFT_MAX_M,
        "force_lateral_drift_penalty": force_lateral_drift_penalty,
        "force_lateral_drift_success_rate": force_lateral_drift_success_rate,
        "straightness_success_rate": straightness_success_rate,
        "best_eligible": 1.0 if best_eligible else 0.0,
    }
    aggregate.update(aggregate_force_compliance_v2(episodes))
    aggregate.update({
        f"mean_{name}": _mean([episode[name] for episode in episodes])
        for name in CONTACT_BALANCE_KEYS
    })
    aggregate.update({
        f"mean_{name}": _mean([episode[name] for episode in episodes])
        for name in ACTION_BALANCE_KEYS
    })
    aggregate.update(
        {
            "force_recovery_success_rate": aggregate["external_force_recovery_success_rate"],
            "force_recovery_fall_rate": aggregate["fall_rate"],
            "force_recovery_post_force_mean_forward_velocity": aggregate[
                "post_force_mean_forward_velocity"
            ],
            "force_recovery_post_force_mean_abs_lateral_velocity": aggregate[
                "post_force_mean_abs_lateral_velocity"
            ],
            "force_recovery_post_force_mean_abs_yaw_rate": aggregate[
                "post_force_mean_abs_yaw_rate"
            ],
            "force_recovery_mean_abs_final_lateral_displacement": aggregate[
                "mean_abs_final_lateral_displacement"
            ],
        }
    )
    return aggregate


def _mean(values: list[float | int | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / max(len(clean), 1)


def _mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _max_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _vector_norm(values: Any) -> float | None:
    if values is None:
        return None
    try:
        clean = [float(value) for value in values]
    except TypeError:
        return None
    if not clean:
        return None
    return math.sqrt(sum(value * value for value in clean))


def _vector_delta_norm(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except TypeError:
        return None
    if not left_values or len(left_values) != len(right_values):
        return None
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left_values, right_values)))


def _contact_imbalance(contact_ratio: dict[str, float]) -> float:
    values = list(contact_ratio.values())
    return max(values) - min(values) if values else 0.0


def initialize_policy_from_checkpoint(*, model: Any, checkpoint_path: str | Path, device: str) -> dict[str, Any]:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    source = PPO.load(str(checkpoint_path), device=device)
    source_state = source.policy.state_dict()
    target_state = model.policy.state_dict()
    copied: list[str] = []
    partially_copied: list[str] = []
    skipped: list[str] = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped.append(key)
            continue
        source_tensor = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            target_state[key] = source_tensor.clone()
            copied.append(key)
            continue
        if len(source_tensor.shape) == 2 and len(target_tensor.shape) == 2:
            rows = min(source_tensor.shape[0], target_tensor.shape[0])
            cols = min(source_tensor.shape[1], target_tensor.shape[1])
            if rows > 0 and cols > 0:
                patched = target_tensor.clone()
                if source_tensor.shape[1] != target_tensor.shape[1]:
                    patched.zero_()
                patched[:rows, :cols] = source_tensor[:rows, :cols]
                target_state[key] = patched
                partially_copied.append(key)
                continue
        skipped.append(key)

    model.policy.load_state_dict(target_state)
    return {
        "checkpoint": str(checkpoint_path),
        "copied": copied,
        "partially_copied": partially_copied,
        "skipped": skipped,
    }
