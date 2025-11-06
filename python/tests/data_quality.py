from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

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
                           thresholds: Union[Sequence[float], float],
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
            # Missing and illed per person instead of per feature
            frame_missing.append(missing_matrix.any(axis=1).tolist())
            frame_illed.append(illed_matrix.any(axis=1).tolist())
        else:
            frame_missing.append([False] * len(PersonSkeleton.FEATURE_ORDER))
            frame_illed.append([False] * len(PersonSkeleton.FEATURE_ORDER))

    out = df.copy()
    out["person_skeleton_quality"] = records
    out["frame_missing_features"] = frame_missing
    out["frame_illed_features"] = frame_illed
    return out

__all__ = [
    "evaluate_frame_skeletons",
    "annotate_frame_quality",
]

if __name__ == "__main__":
    df = pd.read_pickle("../data/export/data.pkl")
    df = annotate_frame_quality(df, 100)
    kp_missing = [1 if any(df.iloc[k]['frame_missing_features']) else 0 for k in range(len(df))]
    kp_illed = [1 if any(df.iloc[k]['frame_illed_features']) else 0 for k in range(len(df))]
    c = 9
