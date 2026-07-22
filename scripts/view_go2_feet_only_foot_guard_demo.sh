#!/usr/bin/env bash
set -euo pipefail

ONNX_POLICY="${ONNX_POLICY:-external/reference_policies/unitree_go2_velocity_flat/policy.onnx}"
MODEL="${MODEL:-external/unitree_mujoco/unitree_robots/go2/flat_scene.xml}"
MAX_MOUSE_FORCE="${MAX_MOUSE_FORCE:-30}"
TARGET_FORWARD_VELOCITY="${TARGET_FORWARD_VELOCITY:-0.40}"

.venv/bin/mjpython -m go2_mini_lab.mujoco_viewer \
  --task velocity_flat \
  --model "$MODEL" \
  --fixed-command \
  --target-forward-velocity "$TARGET_FORWARD_VELOCITY" \
  --target-lateral-velocity 0.0 \
  --target-yaw-rate 0.0 \
  --observation-mode policy \
  --policy-action-mode onnx_safety_layer \
  --residual-onnx-policy "$ONNX_POLICY" \
  --safety-layer-action-override 0 0 0 0 \
  --force-safety-trigger-source oracle \
  --force-response-router-mode semantic_oracle \
  --force-response-profile body_yield_foot_guard \
  --force-response-foot-kp-scale 1.05 \
  --force-response-foot-kd-scale 1.15 \
  --force-impedance-mode onset \
  --force-impedance-joint-scope stance_hip_calf \
  --force-impedance-kp-scale 1.05 \
  --force-impedance-kd-scale 1.15 \
  --force-impedance-delay-s 0.0 \
  --force-impedance-hold-s 0.12 \
  --force-impedance-recovery-s 0.08 \
  --force-reference-governor-mode off \
  --fullbody-reference-mode bounded_compliant \
  --velocity-reward-profile gentle_fullbody_teacher_push_recovery \
  --velocity-command-frame world \
  --velocity-pose-profile official \
  --policy-leg-order mjcf \
  --action-scale 0.5 \
  --action-smoothing 0.15 \
  --reset-settle 0.0 \
  --pd-kp 20 \
  --pd-kd 1 \
  --torque-limit 23.5 \
  --deterministic \
  --max-mouse-force "$MAX_MOUSE_FORCE"
