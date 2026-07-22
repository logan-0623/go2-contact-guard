from __future__ import annotations

import argparse
from typing import Sequence

from .rl_env import (
    EXTERNAL_FORCE_DIRECTION_MODES,
    EXTERNAL_FORCE_MODES,
    FORCE_RESPONSE_PROFILES,
    FORCE_RESPONSE_ROUTER_MODES,
    FORCE_SAFETY_TRIGGER_SOURCES,
    FULLBODY_REFERENCE_MODES,
    OBSERVATION_MODES,
    POLICY_ACTION_MODES,
    POLICY_LEG_ORDER_PROFILES,
    VELOCITY_COMMAND_FRAMES,
    VELOCITY_REWARD_PROFILES,
)


BASE_EXTERNAL_FORCE_BODIES = ("base_link",)
THIGH_EXTERNAL_FORCE_BODIES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
CALF_EXTERNAL_FORCE_BODIES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
FOOT_EXTERNAL_FORCE_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
EXTERNAL_FORCE_BODY_PROFILES = {
    "base_only": BASE_EXTERNAL_FORCE_BODIES,
    "base_thighs": BASE_EXTERNAL_FORCE_BODIES + THIGH_EXTERNAL_FORCE_BODIES,
    "legs": THIGH_EXTERNAL_FORCE_BODIES + CALF_EXTERNAL_FORCE_BODIES,
    "legs_with_feet": THIGH_EXTERNAL_FORCE_BODIES + CALF_EXTERNAL_FORCE_BODIES + FOOT_EXTERNAL_FORCE_BODIES,
    "feet_only": FOOT_EXTERNAL_FORCE_BODIES,
    "all": BASE_EXTERNAL_FORCE_BODIES + THIGH_EXTERNAL_FORCE_BODIES + CALF_EXTERNAL_FORCE_BODIES,
    "all_with_feet": (
        BASE_EXTERNAL_FORCE_BODIES
        + THIGH_EXTERNAL_FORCE_BODIES
        + CALF_EXTERNAL_FORCE_BODIES
        + FOOT_EXTERNAL_FORCE_BODIES
    ),
}


def add_velocity_reward_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--velocity-reward-profile",
        choices=sorted(VELOCITY_REWARD_PROFILES),
        default="default",
        help="Reward weight preset for velocity tasks.",
    )
    parser.add_argument(
        "--velocity-command-frame",
        choices=VELOCITY_COMMAND_FRAMES,
        default="world",
        help=(
            "Frame used for xy velocity tracking rewards. "
            "body tracks commands in the robot yaw frame."
        ),
    )
    parser.add_argument(
        "--reward-term",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override one velocity reward weight, for example "
            "--reward-term velocity_shortfall_penalty=-20. "
            "May be passed multiple times."
        ),
    )


def add_policy_leg_order_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-leg-order",
        choices=sorted(POLICY_LEG_ORDER_PROFILES),
        default="actuator",
        help=(
            "Leg order used in policy observation/action joint blocks. "
            "Use mjcf for the exported ONNX reference policy."
        ),
    )


def add_observation_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="policy",
        help=(
            "Actor observation vocabulary. policy is deployable proprioception; "
            "privileged adds simulator-only state for teacher training."
        ),
    )


def add_policy_action_mode_args(
    parser: argparse.ArgumentParser,
    *,
    onnx_policy_arg: str = "--onnx-policy",
    onnx_policy_dest: str = "onnx_policy",
    onnx_normalizer_arg: str = "--onnx-normalizer-checkpoint",
    onnx_normalizer_dest: str = "onnx_normalizer_checkpoint",
) -> None:
    parser.add_argument(
        "--policy-action-mode",
        choices=POLICY_ACTION_MODES,
        default="full_action",
        help=(
            "How environment actions are interpreted. full_action applies PPO actions directly; "
            "onnx_residual applies frozen ONNX action plus scaled PPO residual; "
            "onnx_safety_layer uses a low-dimensional PPO action to tune event-triggered safety layers."
        ),
    )
    parser.add_argument(
        onnx_policy_arg,
        dest=onnx_policy_dest,
        help="Frozen ONNX policy used as the base action for ONNX-backed action modes.",
    )
    parser.add_argument(
        onnx_normalizer_arg,
        dest=onnx_normalizer_dest,
        help="Optional actor observation normalizer checkpoint for the residual base ONNX policy.",
    )
    parser.add_argument(
        "--residual-action-scale",
        type=float,
        default=0.0,
        help="Scale applied to PPO residual actions in onnx_residual mode.",
    )


def add_fullbody_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fullbody-reference-mode",
        choices=FULLBODY_REFERENCE_MODES,
        default="static",
        help="Full-body tracking reference generator.",
    )
    parser.add_argument("--gait-frequency", type=float, help="Phase-trot reference frequency in Hz.")
    parser.add_argument("--gait-step-length", type=float, help="Phase-trot foot x amplitude in meters.")
    parser.add_argument("--gait-swing-height", type=float, help="Phase-trot swing height in meters.")
    parser.add_argument("--gait-thigh-amplitude", type=float, help="Phase-trot thigh joint amplitude in radians.")
    parser.add_argument("--gait-calf-amplitude", type=float, help="Phase-trot calf joint amplitude in radians.")
    parser.add_argument(
        "--nominal-reference-dataset",
        help="Phase-conditioned rollout reference .npz used by rollout full-body reference modes.",
    )


def add_force_safety_trigger_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force-safety-trigger-source",
        choices=FORCE_SAFETY_TRIGGER_SOURCES,
        default="oracle",
        help=(
            "Source used to trigger transient force-safety layers. oracle keeps legacy "
            "sim-force schedule behavior; deployable uses proprioceptive onset detection; "
            "deployable_v2 uses residual acceleration with a short-term baseline; "
            "deployable_history uses a trained proprioceptive history estimator; "
            "deployable_hybrid requires both history-estimator and residual-acceleration evidence; "
            "deployable_history_or_v2 accepts either history-estimator or residual-acceleration evidence; "
            "deployable_history_gated_v2 accepts history-estimator evidence or gated residual-acceleration evidence; "
            "oracle_or_deployable enables either legacy source; oracle_or_deployable_v2 "
            "combines oracle and deployable_v2; oracle_or_deployable_history combines "
            "oracle and the history estimator; oracle_or_deployable_hybrid combines oracle "
            "and hybrid deployable detection; oracle_or_deployable_history_or_v2 combines "
            "oracle and deployable_history_or_v2; oracle_or_deployable_history_gated_v2 combines "
            "oracle and deployable_history_gated_v2."
        ),
    )
    parser.add_argument(
        "--force-safety-history-estimator",
        help="Path to a trained history event estimator JSON for history-based force-safety trigger sources.",
    )
    parser.add_argument(
        "--force-safety-detector-linear-acceleration-threshold",
        type=float,
        default=1.5,
        help="Deployable safety detector linear base acceleration threshold in m/s^2. 0 disables this signal.",
    )
    parser.add_argument(
        "--force-safety-detector-angular-acceleration-threshold",
        type=float,
        default=8.0,
        help="Deployable safety detector angular base acceleration threshold in rad/s^2. 0 disables this signal.",
    )
    parser.add_argument(
        "--force-safety-detector-joint-error-threshold",
        type=float,
        default=0.20,
        help="Deployable safety detector joint tracking error threshold in radians. 0 disables this signal.",
    )
    parser.add_argument(
        "--force-safety-detector-joint-velocity-threshold",
        type=float,
        default=12.0,
        help="Deployable safety detector joint velocity threshold in rad/s. 0 disables this signal.",
    )
    parser.add_argument(
        "--force-safety-detector-contact-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable contact-loss trigger in the deployable force-safety detector.",
    )
    parser.add_argument(
        "--force-safety-detector-enable-after-s",
        type=float,
        default=0.0,
        help="Ignore deployable safety-detector triggers before this episode time in seconds.",
    )
    parser.add_argument(
        "--force-safety-detector-hold-s",
        type=float,
        default=0.25,
        help="Duration of the deployable detector safety window after onset.",
    )
    parser.add_argument(
        "--force-safety-detector-recovery-s",
        type=float,
        default=0.12,
        help="Diagnostic gate recovery duration after the deployable detector hold window.",
    )


def add_force_response_router_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force-response-router-mode",
        choices=FORCE_RESPONSE_ROUTER_MODES,
        default="off",
        help=(
            "Semantic response selector for force-safety layers. off preserves legacy scalar behavior; "
            "semantic_oracle routes by simulator force body; deployable_semantic uses deployable triggers "
            "without simulator force-body labels."
        ),
    )
    parser.add_argument(
        "--force-response-profile",
        choices=FORCE_RESPONSE_PROFILES,
        default="body_yield_foot_guard",
        help="Named mapping from disturbed body class to impedance/governor response.",
    )
    parser.add_argument(
        "--force-response-foot-kp-scale",
        type=float,
        default=1.05,
        help="Kp multiplier for stance-foot support guard windows.",
    )
    parser.add_argument(
        "--force-response-foot-kd-scale",
        type=float,
        default=1.15,
        help="Kd multiplier for foot support/restep guard windows.",
    )


def add_external_force_args(parser: argparse.ArgumentParser, *, include_curriculum: bool = False) -> None:
    parser.add_argument(
        "--external-force-mode",
        choices=EXTERNAL_FORCE_MODES,
        default="constant",
        help="External force generator. spring uses spring-anchor dynamics.",
    )
    parser.add_argument(
        "--external-force-probability",
        type=float,
        default=0.0,
        help="Per-episode probability of applying a random MuJoCo xfrc_applied force.",
    )
    parser.add_argument(
        "--external-force-body",
        action="append",
        default=[],
        metavar="BODY",
        help=(
            "Body name that may receive random external force. "
            "May be passed multiple times. Overrides --external-force-body-profile."
        ),
    )
    parser.add_argument(
        "--external-force-body-profile",
        choices=sorted(EXTERNAL_FORCE_BODY_PROFILES),
        default="base_only",
        help=(
            "Named disturbance body set used when --external-force-body is not passed. "
            "all preserves the historical base/thigh/calf profile; all_with_feet also includes feet."
        ),
    )
    parser.add_argument(
        "--external-force-active-body-count",
        type=int,
        default=1,
        help="Number of configured bodies to perturb together in spring mode.",
    )
    parser.add_argument(
        "--external-force-events-min",
        type=int,
        default=1,
        help=(
            "Minimum number of disturbance events in an episode when the per-episode "
            "external force probability is sampled as active."
        ),
    )
    parser.add_argument(
        "--external-force-events-max",
        type=int,
        default=1,
        help=(
            "Maximum number of disturbance events in an episode when the per-episode "
            "external force probability is sampled as active."
        ),
    )
    parser.add_argument(
        "--external-force-rest-min",
        type=float,
        default=1.0,
        help="Minimum rest time between disturbance events, in seconds.",
    )
    parser.add_argument(
        "--external-force-rest-max",
        type=float,
        default=2.0,
        help="Maximum rest time between disturbance events, in seconds.",
    )
    parser.add_argument(
        "--external-force-min",
        type=float,
        default=0.0,
        help="Minimum sampled external force magnitude in Newtons.",
    )
    parser.add_argument(
        "--external-force-max",
        type=float,
        default=0.0,
        help="Maximum sampled external force magnitude in Newtons. 0 disables random external force.",
    )
    parser.add_argument(
        "--external-force-start-min",
        type=float,
        default=1.0,
        help="Earliest external force start time in each episode, in seconds.",
    )
    parser.add_argument(
        "--external-force-start-max",
        type=float,
        default=3.0,
        help="Latest external force start time in each episode, in seconds.",
    )
    parser.add_argument(
        "--external-force-duration-min",
        type=float,
        default=0.1,
        help="Minimum external force duration in seconds.",
    )
    parser.add_argument(
        "--external-force-duration-max",
        type=float,
        default=0.3,
        help="Maximum external force duration in seconds.",
    )
    parser.add_argument(
        "--external-force-z-fraction",
        type=float,
        default=0.0,
        help="Relative vertical component when sampling force direction. 0 means horizontal-only pushes.",
    )
    parser.add_argument(
        "--external-force-direction-angle",
        type=float,
        help=(
            "Fixed horizontal force direction angle in radians from +x toward +y. "
            "Omit to sample directions randomly."
        ),
    )
    parser.add_argument(
        "--external-force-direction-mode",
        choices=EXTERNAL_FORCE_DIRECTION_MODES,
        default="default",
        help=(
            "Direction sampler used when --external-force-direction-angle is omitted. "
            "lateral_bimodal samples +/-90 degree side pushes; lateral_mixed biases toward them; "
            "yielding_tail_mixed focuses backward, rear-diagonal, and side pushes."
        ),
    )
    parser.add_argument(
        "--external-force-lateral-probability",
        type=float,
        default=0.85,
        help="Side-push probability used by --external-force-direction-mode lateral_mixed.",
    )
    parser.add_argument(
        "--external-force-torque-max",
        type=float,
        default=0.0,
        help="Maximum direct random body torque disturbance in N*m. 0 disables torque disturbance.",
    )
    parser.add_argument(
        "--external-force-spring-stiffness-min",
        type=float,
        default=50.0,
        help="Minimum spring stiffness for spring mode, in N/m.",
    )
    parser.add_argument(
        "--external-force-spring-stiffness-max",
        type=float,
        default=250.0,
        help="Maximum spring stiffness for spring mode, in N/m.",
    )
    parser.add_argument(
        "--external-force-spring-damping",
        type=float,
        default=2.0,
        help="Interaction damping for spring mode.",
    )
    parser.add_argument(
        "--external-force-guiding-probability",
        type=float,
        default=0.5,
        help="Probability that a spring-mode sample uses a moving guiding anchor instead of resistive contact.",
    )
    parser.add_argument(
        "--external-force-transition",
        type=float,
        default=0.08,
        help="Ramp-in/ramp-out time for spring forces, in seconds.",
    )
    parser.add_argument(
        "--external-force-net-force-limit",
        type=float,
        default=0.0,
        help="Optional net force cap across active spring bodies in Newtons. 0 disables the extra cap.",
    )
    parser.add_argument(
        "--external-force-net-torque-limit",
        type=float,
        default=0.0,
        help="Optional net torque cap across active spring bodies in N*m. 0 disables the extra cap.",
    )
    parser.add_argument(
        "--external-force-reference-mass",
        type=float,
        help="Admittance reference mass for moving guiding spring anchors.",
    )
    parser.add_argument(
        "--external-force-reference-damping",
        type=float,
        help="Extra damping for moving guiding spring anchors.",
    )
    parser.add_argument(
        "--external-force-reference-velocity-clip",
        type=float,
        help="Maximum guiding-anchor reference velocity in m/s.",
    )
    parser.add_argument(
        "--external-force-reference-acceleration-clip",
        type=float,
        help="Maximum guiding-anchor reference acceleration in m/s^2.",
    )
    parser.add_argument(
        "--external-force-safe-limit-min",
        type=float,
        default=0.0,
        help="Minimum episode-level compliant driving-force limit in Newtons. 0 disables the command.",
    )
    parser.add_argument(
        "--external-force-safe-limit-max",
        type=float,
        default=0.0,
        help="Maximum episode-level compliant driving-force limit in Newtons. 0 disables the command.",
    )
    parser.add_argument(
        "--external-force-safe-margin",
        type=float,
        default=10.0,
        help="Allowed applied-force margin above the sampled safe limit in Newtons.",
    )
    parser.add_argument(
        "--external-force-observation",
        action="store_true",
        help=(
            "Append normalized disturbance phase, force, and torque signals to velocity-task observations. "
            "Use this for BC-initialized compliance policies with a larger observation size."
        ),
    )
    if include_curriculum:
        parser.add_argument(
            "--external-force-curriculum-start",
            type=float,
            help=(
                "Initial maximum external force magnitude in Newtons. "
                "If omitted with --external-force-curriculum-steps > 0, starts at 0 N."
            ),
        )
        parser.add_argument(
            "--external-force-curriculum-steps",
            type=int,
            default=0,
            help="Training timesteps used to ramp curriculum start force to --external-force-max.",
        )


def resolve_policy_leg_order(profile: str) -> tuple[str, ...]:
    try:
        return POLICY_LEG_ORDER_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_LEG_ORDER_PROFILES))
        raise ValueError(f"unknown policy leg order {profile!r}; choose one of: {choices}") from exc


def resolve_external_force_body_names(values: Sequence[str], profile: str = "base_only") -> tuple[str, ...]:
    body_names = tuple(value.strip() for value in values if value.strip())
    if body_names:
        return body_names
    try:
        return EXTERNAL_FORCE_BODY_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(EXTERNAL_FORCE_BODY_PROFILES))
        raise ValueError(f"unknown external force body profile {profile!r}; choose one of: {choices}") from exc


def parse_reward_overrides(values: Sequence[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"reward override must be NAME=VALUE, got {value!r}")
        name, raw_number = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"reward override has an empty name: {value!r}")
        try:
            overrides[name] = float(raw_number)
        except ValueError as exc:
            raise ValueError(f"reward override {name!r} must be a float, got {raw_number!r}") from exc
    return overrides
