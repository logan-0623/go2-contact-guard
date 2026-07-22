from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def recommended_z_safe_from_file(path: str | Path, *, margin: float = 0.02) -> float:
    with np.load(Path(path), allow_pickle=False) as data:
        if "base_height_ref" not in data.files:
            raise ValueError(f"reference dataset has no base_height_ref: {path}")
        base_height = np.asarray(data["base_height_ref"], dtype=np.float32)
    if base_height.size == 0:
        raise ValueError(f"reference dataset has empty base_height_ref: {path}")
    return float(np.percentile(base_height, 5.0) - float(margin))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print reference-derived anti-collapse z_safe.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--margin", type=float, default=0.02)
    args = parser.parse_args(argv)

    print(f"{recommended_z_safe_from_file(args.dataset, margin=args.margin):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
