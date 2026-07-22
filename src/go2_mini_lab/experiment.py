from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence


@dataclass(frozen=True)
class ExperimentManifest:
    name: str
    script: Path
    env: dict[str, str]
    args: tuple[str, ...] = ()
    description: str = ""


def load_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"experiment manifest must be a JSON object: {manifest_path}")

    script = data.get("script")
    if not isinstance(script, str) or not script:
        raise ValueError(f"experiment manifest requires a non-empty script: {manifest_path}")

    name = data.get("name") or manifest_path.stem
    if not isinstance(name, str):
        raise ValueError(f"experiment manifest name must be a string: {manifest_path}")

    env_data = data.get("env") or {}
    if not isinstance(env_data, dict):
        raise ValueError(f"experiment manifest env must be an object: {manifest_path}")
    env = {str(key): _env_value(value) for key, value in env_data.items()}

    args_data = data.get("args") or []
    if not isinstance(args_data, list):
        raise ValueError(f"experiment manifest args must be a list: {manifest_path}")
    args = tuple(str(value) for value in args_data)

    description = data.get("description") or ""
    if not isinstance(description, str):
        raise ValueError(f"experiment manifest description must be a string: {manifest_path}")

    return ExperimentManifest(
        name=name,
        script=Path(script),
        env=env,
        args=args,
        description=description,
    )


def build_manifest_command(manifest: ExperimentManifest) -> tuple[str, ...]:
    return ("bash", str(manifest.script), *manifest.args)


def format_dry_run(manifest: ExperimentManifest) -> str:
    lines = [f"Experiment: {manifest.name}"]
    if manifest.description:
        lines.append(f"Description: {manifest.description}")
    if manifest.env:
        lines.append("Environment:")
        for key, value in sorted(manifest.env.items()):
            lines.append(f"  {key}={value}")
    lines.append("Command:")
    lines.append(f"  {shlex.join(build_manifest_command(manifest))}")
    return "\n".join(lines)


def run_manifest(
    manifest: ExperimentManifest,
    *,
    dry_run: bool = False,
    cwd: str | Path | None = None,
) -> int:
    if dry_run:
        print(format_dry_run(manifest))
        return 0

    env = os.environ.copy()
    env.update(manifest.env)
    completed = subprocess.run(
        build_manifest_command(manifest),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Go2 experiment manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Print an experiment manifest command.")
    show_parser.add_argument("manifest", type=Path)

    run_parser = subparsers.add_parser("run", help="Run an experiment manifest.")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "show":
        print(format_dry_run(manifest))
        return 0
    if args.command == "run":
        return run_manifest(manifest, dry_run=bool(args.dry_run))
    raise AssertionError(f"unhandled command: {args.command}")


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
