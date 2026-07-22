from pathlib import Path
import math
import unittest

import numpy as np

from go2_mini_lab.record_long_contact_demo import _contact_label, _force_vector_xy, _window_scale


ROOT = Path(__file__).resolve().parents[1]


class LongContactDemoRecordingTests(unittest.TestCase):
    def test_window_scale_supports_three_second_contact_with_soft_edges(self) -> None:
        self.assertEqual(_window_scale(1.99, start_s=2.0, duration_s=3.0, transition_s=0.2), 0.0)
        self.assertAlmostEqual(_window_scale(2.10, start_s=2.0, duration_s=3.0, transition_s=0.2), 0.5)
        self.assertEqual(_window_scale(2.30, start_s=2.0, duration_s=3.0, transition_s=0.2), 1.0)
        self.assertEqual(_window_scale(4.80, start_s=2.0, duration_s=3.0, transition_s=0.2), 1.0)
        self.assertAlmostEqual(_window_scale(4.90, start_s=2.0, duration_s=3.0, transition_s=0.2), 0.5)
        self.assertEqual(_window_scale(5.01, start_s=2.0, duration_s=3.0, transition_s=0.2), 0.0)

    def test_force_vector_uses_horizontal_world_angle(self) -> None:
        np.testing.assert_allclose(_force_vector_xy(30.0, 0.0), [30.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(_force_vector_xy(20.0, math.pi / 2.0), [0.0, 20.0, 0.0], atol=1e-9)

    def test_contact_label_distinguishes_body_leg_and_foot_contact(self) -> None:
        self.assertEqual(_contact_label("base_link"), "body contact")
        self.assertEqual(_contact_label("FR_thigh"), "leg contact")
        self.assertEqual(_contact_label("FR_foot"), "foot contact")

    def test_render_script_records_feet_only_foot_guard_github_asset(self) -> None:
        script = (ROOT / "scripts/render_go2_long_contact_foot_guard_demo.sh").read_text()

        self.assertIn(".venv/bin/mjpython", script)
        self.assertIn("-m go2_mini_lab.record_long_contact_demo", script)
        self.assertIn("docs/assets/go2_long_contact_foot_guard.gif", script)
        self.assertIn("--force-body \"${FORCE_BODY:-FR_foot}\"", script)
        self.assertIn("--force-n \"${FORCE_N:-22}\"", script)
        self.assertIn("--force-direction-angle \"${FORCE_DIRECTION_ANGLE:-1.5708}\"", script)
        self.assertIn("--contact-duration \"${CONTACT_DURATION:-3.0}\"", script)
        self.assertIn("--policy-action-mode onnx_safety_layer", script)
        self.assertIn("--force-response-router-mode semantic_oracle", script)
        self.assertIn("--force-response-profile body_yield_foot_guard", script)
        self.assertIn("--force-reference-governor-mode off", script)
        self.assertIn("--demo-title \"Go2 semantic foot guard\"", script)

    def test_render_script_records_body_guard_github_asset(self) -> None:
        script = (ROOT / "scripts/render_go2_long_contact_body_guard_demo.sh").read_text()

        self.assertIn(".venv/bin/mjpython", script)
        self.assertIn("-m go2_mini_lab.record_long_contact_demo", script)
        self.assertIn("docs/assets/go2_long_contact_body_guard.gif", script)
        self.assertIn("--force-body \"${FORCE_BODY:-base_link}\"", script)
        self.assertIn("--force-n \"${FORCE_N:-22}\"", script)
        self.assertIn("--policy-action-mode onnx_safety_layer", script)
        self.assertIn("--force-response-router-mode semantic_oracle", script)
        self.assertIn("--force-response-profile body_yield_foot_guard", script)
        self.assertIn("--force-reference-governor-mode onset", script)
        self.assertIn("--demo-title \"Go2 long-contact body guard\"", script)

    def test_combined_github_demo_script_runs_body_and_foot_renders(self) -> None:
        script = (ROOT / "scripts/render_go2_long_contact_github_demos.sh").read_text()

        self.assertIn("render_go2_long_contact_body_guard_demo.sh", script)
        self.assertIn("render_go2_long_contact_foot_guard_demo.sh", script)

    def test_gitignore_excludes_experiment_outputs_but_keeps_docs_assets(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text()

        self.assertIn("/reports/", gitignore)
        self.assertIn("/checkpoints/", gitignore)
        self.assertIn("/wandb/", gitignore)
        self.assertNotIn("docs/assets/*.gif", gitignore)
        self.assertNotIn("docs/assets/*.mp4", gitignore)

    def test_readme_embeds_long_contact_gifs_and_generation_command(self) -> None:
        readme = (ROOT / "README.md").read_text()

        self.assertIn("docs/assets/go2_long_contact_body_guard.gif", readme)
        self.assertIn("docs/assets/go2_long_contact_foot_guard.gif", readme)
        self.assertIn("bash scripts/render_go2_long_contact_github_demos.sh", readme)
        self.assertIn("3 s long-contact", readme)


if __name__ == "__main__":
    unittest.main()
