# Go2 Contact Guard

Event-triggered contact recovery layers for a frozen Unitree Go2 locomotion policy.

This repository is a compact public showcase of a research prototype. The core idea is simple: keep a strong nominal ONNX walking policy in control during normal locomotion, then activate a lightweight contact guard during sustained external contact and decay back to the nominal gait after release.

## Demos

| Body long-contact guard | Foot long-contact guard |
|---|---|
| ![Go2 body long-contact guard](docs/assets/go2_long_contact_body_guard.gif) | ![Go2 foot long-contact guard](docs/assets/go2_long_contact_foot_guard.gif) |

Both clips are deterministic MuJoCo renders with a 3 s long-contact window. The body demo applies contact to `base_link`; the foot demo applies contact to `FR_foot`. The examples use simulator contact labels for visualization and upper-bound routing, so they should be read as simulation demonstrations rather than real-robot deployment claims.

## Research Motivation

Robust walking recovery and stable long-contact interaction are related but not identical. A nominal quadruped policy may survive pushes while still showing drift, height loss, yaw error, or slow recovery after the contact ends. This project explores a small wrapper around an existing gait policy:

```text
frozen ONNX gait
  + event/contact trigger
  + semantic body/leg/foot response routing
  + impedance and reference-governor safety layer
  -> contact-aware walking recovery
```

The current public demo focuses on walking under sustained body and foot contact. It does not claim that sub-200 ms impact spikes are solved, and it does not claim real-robot deployment without a non-privileged contact detector.

## Quick Start

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[sim,reference]'
```

Fetch the Unitree MuJoCo model:

```bash
bash examples/fetch_unitree_mujoco.sh
```

Regenerate the GitHub demo assets:

```bash
bash scripts/render_go2_long_contact_github_demos.sh
```

Run the interactive MuJoCo viewer:

```bash
bash scripts/view_go2_feet_only_foot_guard_demo.sh
```

In the viewer, select a body or foot and apply force with the MuJoCo perturbation controls.

## Repository Scope

This public repository intentionally keeps only the compact showcase:

```text
src/go2_mini_lab/      simulation environment and contact-guard implementation
scripts/               demo and viewer entry points
docs/assets/           curated demo GIF/MP4 assets
external/reference_*   lightweight reference policy files
examples/              setup helper for Unitree MuJoCo assets
```

Detailed sweep reports, checkpoints, internal experiment notes, and training logs are not included. They are useful for research iteration but too noisy for a public project landing page.
