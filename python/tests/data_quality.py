from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from utils.pose import PersonSkeleton, build_person_skeletons, extract_raw_keypoints

PersonSummary = Dict[str, Any]


def _extract_person_skeletons(row: pd.Series,
                              base_height: float = 170.0) -> List[PersonSkeleton]:
    coords_obj = row.get("spaceCoords")
    feats = row.get("spaceFeat")
    if coords_obj is None:
        return []

    if isinstance(coords_obj, np.ndarray) and coords_obj.ndim == 2:
        person_entries = [coords_obj[i] for i in range(coords_obj.shape[0])]
    else:
        try:
            person_entries = list(coords_obj)
        except TypeError:
            return []

    head_feats = None
    if isinstance(feats, dict):
        head_feats = feats.get("head")

    return build_person_skeletons(person_entries, base_height=base_height, head_feats=head_feats)


def evaluate_frame_skeletons(frame: pd.Series,
                              thresholds: Union[Sequence[float], float],
                              base_height: float = 170.0) -> List[PersonSummary]:
    skeletons = _extract_person_skeletons(frame, base_height=base_height)
    summaries: List[PersonSummary] = []
    for skeleton in skeletons:
        missing_flags = skeleton.missing_features()
        illed_flags = skeleton.illed_features(thresholds)
        summaries.append(
            {
                "index": skeleton.row_index,
                "pid": skeleton.pid,
                "missing": missing_flags,
                "illed": illed_flags,
                "lengths": skeleton.feature_lengths().tolist(),
            }
        )
    return summaries


def annotate_frame_quality(df: pd.DataFrame,
                           thresholds: Union[Sequence[float], float] = 100,
                           base_height: float = 170.0) -> pd.DataFrame:
    records: List[List[PersonSummary]] = []
    frame_missing: List[List[bool]] = []
    frame_illed: List[List[bool]] = []

    for _, row in df.iterrows():
        person_summaries = evaluate_frame_skeletons(row, thresholds, base_height=base_height)
        records.append(person_summaries)
        if person_summaries:
            missing_matrix = np.array([summary["missing"] for summary in person_summaries], dtype=bool)
            illed_matrix = np.array([summary["illed"] for summary in person_summaries], dtype=bool)
            frame_missing.append(missing_matrix.any(axis=0).tolist())
            frame_illed.append(illed_matrix.any(axis=0).tolist())
        else:
            frame_missing.append([False] * len(PersonSkeleton.FEATURE_ORDER))
            frame_illed.append([False] * len(PersonSkeleton.FEATURE_ORDER))

    out = df.copy()
    out["person_skeleton_quality"] = records
    out["frame_missing_features"] = frame_missing
    out["frame_illed_features"] = frame_illed
    out["frame_missing_any"] = [any(flags) for flags in frame_missing]
    out["frame_illed_any"] = [any(flags) for flags in frame_illed]
    out["frame_good"] = ~(out["frame_missing_any"] | out["frame_illed_any"])
    return out


def plot_segment_quality(df: pd.DataFrame,
                         ordering_column: str = "Timestamp",
                         axes: Optional[np.ndarray] = None,
                         figsize: Tuple[float, float] = (12.0, 16.0),
                         columns: int = 2) -> Tuple[Any, np.ndarray, pd.DataFrame]:
    required_cols = {"Cam", "Vid", "Seg", "frame_missing_any", "frame_illed_any"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    if ordering_column not in df.columns:
        ordering_column = "Timestamp" if "Timestamp" in df.columns else df.columns[0]

    grouped = df.sort_values(ordering_column).groupby(["Vid", "Cam", "Seg"], sort=True)

    segment_arrays: List[np.ndarray] = []
    labels: List[str] = []
    stats_rows: List[Dict[str, Any]] = []

    for (vid, cam, seg), group in grouped:
        status = ~(group["frame_missing_any"] | group["frame_illed_any"])
        arr = status.astype(int).to_numpy()
        segment_arrays.append(arr)
        labels.append(f"V{vid}-C{cam}-S{seg}")
        stats_rows.append(
            {
                "Vid": vid,
                "Cam": cam,
                "Seg": seg,
                "good_frames": int(arr.sum()),
                "bad_frames": int(len(arr) - arr.sum()),
                "total_frames": int(len(arr)),
            }
        )

    if not segment_arrays:
        raise ValueError("No segments available for plotting.")

    num_segments = len(segment_arrays)
    columns = max(1, min(columns, num_segments))
    rows = (num_segments + columns - 1) // columns

    if axes is None:
        fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    else:
        fig = axes.flat[0].figure

    cmap = ListedColormap(["lightgray", "crimson", "forestgreen"])
    axes_list = axes.flatten()
    for ax, label, arr, stats in zip(axes_list, labels, segment_arrays, stats_rows):
        data = arr.reshape(1, -1)
        ax.imshow(data, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{label}\nG:{stats['good_frames']} B:{stats['bad_frames']}", fontsize=8)

    for idx in range(len(segment_arrays), len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle("Segment Quality Spectrum (Green=Good, Red=Bad)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    stats_df = pd.DataFrame(stats_rows)
    return fig, axes, stats_df

def skeleton_sanity(img_size=[960, 540]):
    return 1


def plot_all_segments_dense(json_parent: Path,
                            output_dir: Path,
                            segment_length: int = 600,
                            columns: int = 2)-> List[Path]:
    """
    Iterate over pose JSON files and render dense 600-frame spectrum plots per file.
    Each JSON source produces its own saved image so the output count matches the
    number of iterated files.
    """
    if segment_length <= 0:
        raise ValueError("segment_length must be positive.")
    if columns <= 0:
        raise ValueError("columns must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_parent.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {json_parent}")

    generated_paths: List[Path] = []
    cmap = ListedColormap(["lightgray", "crimson", "forestgreen"])

    for json_path in json_files:
        with open(json_path, "r") as file:
            raw_annotation = json.load(file)
        skeletons = raw_annotation.get("annotations", {}).get("skeletons", [])

        frame_good_flags: List[int] = []
        for idx in range(len(skeletons)):
            frame_coords = extract_raw_keypoints(skeletons, idx)
            if frame_coords is None:
                raise ValueError
            else:
                kp_good: List[bool] = []
                for person_kps in frame_coords.values():
                    if person_kps is None:
                        kp_good.append(False)
                        continue
                    not_nan = np.all(~np.isnan(person_kps))
                    not_illed = skeleton_sanity(person_kps)
                    kp_good.append(bool(not_nan and not_illed))
                frame_good = all(kp_good)
            frame_good_flags.append(1 if frame_good else 0)

        if not frame_good_flags:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No frame data available", ha="center", va="center")
            ax.axis("off")
            fig_path = output_dir / f"{json_path.stem}_segments.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            generated_paths.append(fig_path)
            continue

        segments = [
            frame_good_flags[i:i + segment_length]
            for i in range(0, len(frame_good_flags), segment_length)
        ]
        num_segments = len(segments)
        num_columns = max(1, columns)
        rows = (num_segments + num_columns - 1) // num_columns
        fig_width = max(6, num_columns * 5)
        fig_height = max(2, rows * 1.5)
        fig, axes = plt.subplots(rows, num_columns, figsize=(fig_width, fig_height), squeeze=False)
        axes_list = axes.flatten()

        for seg_idx, (segment, ax) in enumerate(zip(segments, axes_list)):
            arr = np.asarray(segment, dtype=int)
            data = arr.reshape(1, -1)
            start = seg_idx * segment_length
            end = min(len(frame_good_flags), start + segment_length)
            good_count = int(arr.sum())
            bad_count = int(arr.size - good_count)
            label = f"{json_path.stem}-S{seg_idx + 1}"
            ax.imshow(data, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(
                f"{label}\nFrames {start}-{end - 1}\nG:{good_count} B:{bad_count}",
                fontsize=8,
            )

        for idx in range(num_segments, len(axes_list)):
            axes_list[idx].axis("off")

        fig.suptitle(f"{json_path.stem} Frame Quality Spectrum", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fig_path = output_dir / f"{json_path.stem}_segments.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        generated_paths.append(fig_path)

    return generated_paths

__all__ = [
    "evaluate_frame_skeletons",
    "annotate_frame_quality",
    "plot_segment_quality",
]

if __name__ == "__main__":
    # df = pd.read_pickle("../data/export/data.pkl")
    # df_quality = annotate_frame_quality(df, thresholds=100)
    # fig, axes, stats_df = plot_segment_quality(df_quality, ordering_column="Timestamp")
    # # plt.show()
    # c = 9
    raw_json_path = Path("/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco/")
    write_path = Path("../data/results/segment_quality_dense/")
    plot_all_segments_dense(json_parent=raw_json_path, output_dir=write_path)
    c = 9
