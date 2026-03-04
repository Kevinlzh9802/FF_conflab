from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CLUES = ("head", "shoulder", "hip", "foot")


def _coerce_person_ids(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(np.int64).to_numpy()

    # Stable fallback when IDs are non-numeric.
    cats = pd.Categorical(series.astype(str))
    return (cats.codes + 1).astype(np.int64)


def _pack_feat(person_ids: np.ndarray, xy: np.ndarray, orient: np.ndarray) -> np.ndarray:
    if xy.size == 0:
        return np.zeros((0, 4), dtype=float)
    return np.column_stack([person_ids, xy[:, 0], xy[:, 1], orient]).astype(float)


def build_data_kp_from_ground_csv(
    csv_path: str | Path,
    cam: int = 1,
    vid: int = 1,
    seg: int = 1,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {
        "frame_name",
        "obj_id",
        "head_x",
        "head_y",
        "head_orient_rad",
        "shoulder_left_x",
        "shoulder_left_y",
        "shoulder_right_x",
        "shoulder_right_y",
        "shoulder_orient_rad",
        "hip_left_x",
        "hip_left_y",
        "hip_right_x",
        "hip_right_y",
        "hip_orient_rad",
        "foot_left_x",
        "foot_left_y",
        "foot_right_x",
        "foot_right_y",
        "foot_orient_rad",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(missing)}"
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "Cam",
                "Vid",
                "Seg",
                "Timestamp",
                "concat_ts",
                "pixelCoords",
                "spaceCoords",
                "pixelFeat",
                "spaceFeat",
                "GT",
            ]
        )

    frame_rows: List[Dict[str, object]] = []
    grouped = df.groupby("frame_name", sort=False)

    for frame_idx, (frame_name, g) in enumerate(grouped, start=1):
        person_ids = _coerce_person_ids(g["obj_id"])

        head_xy = g[["head_x", "head_y"]].to_numpy(dtype=float)
        head_o = g["head_orient_rad"].to_numpy(dtype=float)

        shoulder_xy = np.column_stack(
            [
                (g["shoulder_left_x"].to_numpy(dtype=float) + g["shoulder_right_x"].to_numpy(dtype=float)) / 2.0,
                (g["shoulder_left_y"].to_numpy(dtype=float) + g["shoulder_right_y"].to_numpy(dtype=float)) / 2.0,
            ]
        )
        shoulder_o = g["shoulder_orient_rad"].to_numpy(dtype=float)

        hip_xy = np.column_stack(
            [
                (g["hip_left_x"].to_numpy(dtype=float) + g["hip_right_x"].to_numpy(dtype=float)) / 2.0,
                (g["hip_left_y"].to_numpy(dtype=float) + g["hip_right_y"].to_numpy(dtype=float)) / 2.0,
            ]
        )
        hip_o = g["hip_orient_rad"].to_numpy(dtype=float)

        foot_xy = np.column_stack(
            [
                (g["foot_left_x"].to_numpy(dtype=float) + g["foot_right_x"].to_numpy(dtype=float)) / 2.0,
                (g["foot_left_y"].to_numpy(dtype=float) + g["foot_right_y"].to_numpy(dtype=float)) / 2.0,
            ]
        )
        foot_o = g["foot_orient_rad"].to_numpy(dtype=float)

        space_coords = np.full((len(g), 20), np.nan, dtype=float)
        space_coords[:, 0:2] = head_xy
        space_coords[:, 2:4] = head_xy
        space_coords[:, 4] = g["shoulder_left_x"].to_numpy(dtype=float)
        space_coords[:, 5] = g["shoulder_left_y"].to_numpy(dtype=float)
        space_coords[:, 6] = g["shoulder_right_x"].to_numpy(dtype=float)
        space_coords[:, 7] = g["shoulder_right_y"].to_numpy(dtype=float)
        space_coords[:, 8] = g["hip_left_x"].to_numpy(dtype=float)
        space_coords[:, 9] = g["hip_left_y"].to_numpy(dtype=float)
        space_coords[:, 10] = g["hip_right_x"].to_numpy(dtype=float)
        space_coords[:, 11] = g["hip_right_y"].to_numpy(dtype=float)
        space_coords[:, 12] = g["hip_left_x"].to_numpy(dtype=float)
        space_coords[:, 13] = g["hip_left_y"].to_numpy(dtype=float)
        space_coords[:, 14] = g["foot_right_x"].to_numpy(dtype=float)
        space_coords[:, 15] = g["foot_right_y"].to_numpy(dtype=float)
        space_coords[:, 16] = g["foot_left_x"].to_numpy(dtype=float)
        space_coords[:, 17] = g["foot_left_y"].to_numpy(dtype=float)
        space_coords[:, 18] = g["foot_right_x"].to_numpy(dtype=float)
        space_coords[:, 19] = g["foot_right_y"].to_numpy(dtype=float)

        space_feat = {
            "head": _pack_feat(person_ids, head_xy, head_o),
            "shoulder": _pack_feat(person_ids, shoulder_xy, shoulder_o),
            "hip": _pack_feat(person_ids, hip_xy, hip_o),
            "foot": _pack_feat(person_ids, foot_xy, foot_o),
        }

        frame_rows.append(
            {
                "row_id": frame_idx,
                "Cam": int(cam),
                "Vid": int(vid),
                "Seg": int(seg),
                "Timestamp": frame_idx,
                "concat_ts": frame_idx,
                "pixelCoords": None,
                "spaceCoords": space_coords,
                "pixelFeat": {k: v.copy() for k, v in space_feat.items()},
                "spaceFeat": space_feat,
                "GT": None,
                "frame_name": frame_name,
            }
        )

    return pd.DataFrame(frame_rows)


def convert_ground_csv_to_data_kp(
    csv_path: str | Path,
    output_pkl: str | Path,
    cam: int = 1,
    vid: int = 1,
    seg: int = 1,
) -> pd.DataFrame:
    data_kp = build_data_kp_from_ground_csv(csv_path=csv_path, cam=cam, vid=vid, seg=seg)
    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    data_kp.to_pickle(output_pkl)
    return data_kp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert extracted ground CSV to GCFF data_kp pickle.")
    parser.add_argument("--csv", required=True, type=str, help="Path to ground_plane_info.csv")
    parser.add_argument("--out", required=True, type=str, help="Path to output data_kp .pkl")
    parser.add_argument("--cam", default=4, type=int)
    parser.add_argument("--vid", default=2, type=int)
    parser.add_argument("--seg", default=8, type=int)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert_ground_csv_to_data_kp(
        csv_path=args.csv,
        output_pkl=args.out,
        cam=args.cam,
        vid=args.vid,
        seg=args.seg,
    )
