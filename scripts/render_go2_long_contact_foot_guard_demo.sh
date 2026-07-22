#!/usr/bin/env bash
set -euo pipefail

ONNX_POLICY="${ONNX_POLICY:-external/reference_policies/unitree_go2_velocity_flat/policy.onnx}"
MODEL="${MODEL:-external/unitree_mujoco/unitree_robots/go2/flat_scene.xml}"
OUTPUT_GIF="${OUTPUT_GIF:-docs/assets/go2_long_contact_foot_guard.gif}"
OUTPUT_MP4="${OUTPUT_MP4:-docs/assets/go2_long_contact_foot_guard.mp4}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-docs/assets/go2_long_contact_foot_guard.json}"

.venv/bin/mjpython -m go2_mini_lab.record_long_contact_demo \
  --task velocity_flat \
  --model "$MODEL" \
  --output-gif "$OUTPUT_GIF" \
  --output-mp4 "$OUTPUT_MP4" \
  --summary-output "$SUMMARY_OUTPUT" \
  --duration "${DURATION:-8.0}" \
  --render-fps "${RENDER_FPS:-12}" \
  --width "${WIDTH:-640}" \
  --height "${HEIGHT:-360}" \
  --demo-title "Go2 semantic foot guard" \
  --demo-subtitle "Frozen ONNX gait + event-triggered foot impedance guard" \
  --target-forward-velocity "${TARGET_FORWARD_VELOCITY:-0.40}" \
  --target-lateral-velocity 0.0 \
  --target-yaw-rate 0.0 \
  --observation-mode policy \
  --policy-action-mode onnx_safety_layer \
  --residual-onnx-policy "$ONNX_POLICY" \
  --safety-layer-action-override 0 0 0 0 \
  --force-safety-trigger-source oracle \
  --force-response-router-mode semantic_oracle \
  --force-response-profile body_yield_foot_guard \
  --force-response-foot-kp-scale "${FOOT_KP_SCALE:-1.05}" \
  --force-response-foot-kd-scale "${FOOT_KD_SCALE:-1.15}" \
  --force-impedance-mode onset \
  --force-impedance-joint-scope stance_hip_calf \
  --force-impedance-kp-scale "${FORCE_IMPEDANCE_KP_SCALE:-1.05}" \
  --force-impedance-kd-scale "${FORCE_IMPEDANCE_KD_SCALE:-1.15}" \
  --force-impedance-delay-s 0.0 \
  --force-impedance-hold-s "${FORCE_IMPEDANCE_HOLD_S:-0.12}" \
  --force-impedance-recovery-s "${FORCE_IMPEDANCE_RECOVERY_S:-0.08}" \
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
  --force-body "${FORCE_BODY:-FR_foot}" \
  --force-n "${FORCE_N:-22}" \
  --force-direction-angle "${FORCE_DIRECTION_ANGLE:-1.5708}" \
  --contact-start "${CONTACT_START:-2.0}" \
  --contact-duration "${CONTACT_DURATION:-3.0}" \
  --force-transition "${FORCE_TRANSITION:-0.20}"
