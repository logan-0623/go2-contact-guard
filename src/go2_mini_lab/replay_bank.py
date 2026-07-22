from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def load_trace_rows(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_force_replay_bank(
    rows: Sequence[dict[str, Any]],
    *,
    checkpoint_path: str | None = None,
    vecnormalize_path: str | None = None,
    source_trace: str | None = None,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    previous_by_episode: dict[int, dict[str, Any]] = {}

    for row in rows:
        episode = int(row.get("episode", 0))
        previous = previous_by_episode.get(episode)
        active = bool(row.get("external_force_active", False))
        was_active = bool(previous.get("external_force_active", False)) if previous else False
        if active and not was_active:
            initial = previous or row
            events.append(
                {
                    "episode": episode,
                    "event_index": len(events),
                    "pre_event_step": int(initial.get("step", row.get("step", 0))),
                    "event_start_step": int(row.get("step", 0)),
                    "event_start_s": float(row.get("t", 0.0)),
                    "initial_state": _initial_state_from_row(initial),
                    "force_event": _force_event_from_row(row),
                }
            )
        previous_by_episode[episode] = row

    return {
        "format": "go2_force_replay_bank_v1",
        "source_trace": source_trace,
        "checkpoint": checkpoint_path,
        "vecnormalize": vecnormalize_path,
        "event_count": len(events),
        "controller_ladder": controller_ladder_manifest(),
        "events": events,
    }


def controller_ladder_manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": "onnx_only",
            "description": "Frozen ONNX gait with no residual and no safety layer.",
            "env": {
                "POLICY_ACTION_MODE": "onnx_residual",
                "RESIDUAL_ACTION_SCALE": "0.0",
                "FORCE_IMPEDANCE_MODE": "off",
                "FORCE_REFERENCE_GOVERNOR_MODE": "off",
            },
        },
        {
            "name": "no_safety_layer",
            "description": "Current policy/action mode with safety layers disabled.",
            "env": {
                "FORCE_IMPEDANCE_MODE": "off",
                "FORCE_REFERENCE_GOVERNOR_MODE": "off",
            },
        },
        {
            "name": "oracle_trigger",
            "description": "Event-triggered safety layers using simulator force windows as oracle trigger.",
            "env": {
                "POLICY_ACTION_MODE": "onnx_safety_layer",
                "FORCE_SAFETY_TRIGGER_SOURCE": "oracle",
            },
        },
        {
            "name": "deployable_trigger",
            "description": "Event-triggered safety layers using proprioceptive detector trigger.",
            "env": {
                "POLICY_ACTION_MODE": "onnx_safety_layer",
                "FORCE_SAFETY_TRIGGER_SOURCE": "deployable",
            },
        },
        {
            "name": "max_authority_safety_layer",
            "description": "Oracle-triggered hand-designed max-authority 4D safety-layer action.",
            "env": {
                "POLICY_ACTION_MODE": "onnx_safety_layer",
                "FORCE_SAFETY_TRIGGER_SOURCE": "oracle",
                "SAFETY_LAYER_ACTION_OVERRIDE": "-1.0 1.0 1.0 1.0",
            },
        },
    ]


def _initial_state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "t",
        "qpos",
        "qvel",
        "ctrl",
        "command",
        "last_action",
        "onnx_action",
        "residual_action",
        "safety_layer_action",
        "final_action",
        "applied_action",
        "external_force_safe_limit_n",
        "external_force_schedule",
        "force_reference_governor_offset",
        "force_reference_governor_velocity",
        "force_reference_governor_gate",
    )
    return {key: row[key] for key in keys if key in row}


def _force_event_from_row(row: dict[str, Any]) -> dict[str, Any]:
    schedule = row.get("external_force_schedule") or {"pulses": [], "springs": []}
    return {
        "body": row.get("external_force_body"),
        "direction": row.get("external_force_direction"),
        "vector": row.get("external_force_vector"),
        "magnitude_n": row.get("external_force_magnitude_n"),
        "safe_limit_n": row.get("external_force_safe_limit_n"),
        "safe_margin_n": row.get("external_force_safe_margin_n"),
        "mode": row.get("external_force_mode"),
        "schedule": schedule,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic force-event replay bank from trace JSONL.")
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--vecnormalize")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load_trace_rows(args.trace_jsonl)
    bank = build_force_replay_bank(
        rows,
        checkpoint_path=args.checkpoint,
        vecnormalize_path=args.vecnormalize,
        source_trace=str(args.trace_jsonl),
    )
    _write_json(args.output, bank)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
