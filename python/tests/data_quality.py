from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.pose import PersonSkeleton, build_person_skeletons

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

__all__ = [
    "evaluate_frame_skeletons",
    "annotate_frame_quality",
    "plot_segment_quality",
]

if __name__ == "__main__":
    df = pd.read_pickle("../data/export/data.pkl")
    df_quality = annotate_frame_quality(df, thresholds=100)
    fig, axes, stats_df = plot_segment_quality(df_quality, ordering_column="Timestamp")
    plt.show()
    c = 9
