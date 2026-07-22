from __future__ import annotations

from pathlib import Path
from typing import Any

from .rl_env import Go2StandBalanceEnv, StandBalanceEnvConfig

try:
    import gymnasium as gym
except ImportError:
    gym = None


class Go2StandBalanceGymEnv(gym.Env if gym is not None else object):
    """Gymnasium wrapper around Go2StandBalanceEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        model_path: str | Path = "external/unitree_mujoco/unitree_robots/go2/scene.xml",
        config: StandBalanceEnvConfig | None = None,
    ) -> None:
        try:
            import numpy as np
            from gymnasium import spaces
        except ImportError as exc:
            raise RuntimeError(
                "Training dependencies are missing. Install with: pip install -e '.[train]'"
            ) from exc

        self.np = np
        self.env = Go2StandBalanceEnv(model_path=model_path, config=config)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.env.observation_names),),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        noise = float(options.get("noise", 0.0))
        replay_state = options.get("replay_state")
        obs, info = self.env.reset(seed=seed, noise=noise, replay_state=replay_state)
        return obs.astype(self.np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return obs.astype(self.np.float32), float(reward), bool(terminated), bool(truncated), info

    def set_external_force_curriculum_progress(self, progress: float) -> None:
        self.env.set_external_force_curriculum_progress(progress)

    def close(self) -> None:
        return None
