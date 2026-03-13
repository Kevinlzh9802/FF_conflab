from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

from utils.ground_csv_to_data_kp import convert_ground_csv_to_data_kp


def _parse_cam_vid_seg(path: Path) -> Tuple[int, int, int]:
    stem = path.stem
    if len(stem) != 3 or not stem.isdigit():
        raise ValueError(
            f"Expected a 3-digit filename like '228.csv', got {path.name}"
        )
    return int(stem[0]), int(stem[1]), int(stem[2])


def _segment_key(path: Path) -> int:
    stem = path.stem
    if len(stem) != 3 or not stem.isdigit():
        raise ValueError(
            f"Expected a 3-digit filename like '228.csv', got {path.name}"
        )
    return int(stem)


def _iter_csv_files(input_dir: Path) -> Iterable[Path]:
    csv_files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"]
    return sorted(csv_files, key=_segment_key)


def _normalize_combined_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["row_id"] = np.arange(1, len(out) + 1, dtype=np.int64)
    if "concat_ts" in out.columns:
        out["concat_ts"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def batch_convert_ground_csv_dir(input_dir: str | Path,
                                 output_dir: str | Path,
                                 overwrite: bool = False,
                                 combined_name: str = "data.pkl") -> tuple[list[Path], Path]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(_iter_csv_files(input_path))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_path}")

    written_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    for csv_file in csv_files:
        try:
            cam, vid, seg = _parse_cam_vid_seg(csv_file)
            out_file = output_path / f"{csv_file.stem}.pkl"

            if out_file.exists() and not overwrite:
                data_kp = pd.read_pickle(out_file)
                print(f"Using existing file: {out_file}")
            else:
                data_kp = convert_ground_csv_to_data_kp(
                    csv_path=csv_file,
                    output_pkl=out_file,
                    cam=cam,
                    vid=vid,
                    seg=seg,
                )
                written_paths.append(out_file)
                print(f"Converted {csv_file.name} -> {out_file.name} (cam={cam}, vid={vid}, seg={seg})")

            # Filename is the source of truth for segment identity.
            data_kp = data_kp.copy()
            data_kp["Cam"] = cam
            data_kp["Vid"] = vid
            data_kp["Seg"] = seg
            frames.append(data_kp)
        except Exception as exc:
            print(f"Skipping {csv_file.name}: {exc}")
            continue

    combined_path = output_path / combined_name
    if not frames:
        raise RuntimeError(f"No valid CSV files were converted from: {input_path}")
    combined = _normalize_combined_table(pd.concat(frames, ignore_index=True))
    combined.to_pickle(combined_path)
    print(f"Wrote combined pickle: {combined_path}")

    return written_paths, combined_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert SAM4D ground CSV files into per-segment and combined data_kp pickle files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing source CSV files named like 228.csv, 433.csv, etc.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where converted .pkl files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-segment .pkl files in the output directory.",
    )
    parser.add_argument(
        "--combined-name",
        type=str,
        default="data.pkl",
        help="Filename for the concatenated pickle written under --output-dir.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    batch_convert_ground_csv_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
        combined_name=args.combined_name,
    )
