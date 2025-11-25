from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from utils.pose import PersonSkeleton, build_person_skeletons, extract_raw_keypoints
from utils.data import extract_group_annotations
from utils.constants import RAW_jSON_PATH, GROUP_ANNOTATIONS_PATH, SEGS


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

def frame_quality_person():
    data = {}
    json_files = sorted(RAW_jSON_PATH.glob("*.json"))
    
    for json_path in json_files:

        stem_parts = json_path.stem.split("_")
        cam = vid = seg = None
        for part in stem_parts:
            if part.startswith("cam"):
                cam = int(part.replace("cam", ""))
            elif part.startswith("vid"):
                vid = int(part.replace("vid", ""))
            elif part.startswith("seg"):
                seg = int(part.replace("seg", ""))
        seg_name = f"{cam}{vid}{seg}"

        with open(json_path, "r") as file:
            raw_annotation = json.load(file)
        skeletons = raw_annotation.get("annotations", {}).get("skeletons", [])

        segment_data = {}
        num_frames = len(skeletons)

        for idx in range(num_frames):
            frame_coords = extract_raw_keypoints(skeletons, idx)
            if frame_coords is None:
                continue
            for person_id, keypoints in frame_coords.items():
                if person_id not in segment_data:
                    segment_data[person_id] = np.zeros(num_frames, dtype=int)
                person_quality = not np.isnan(keypoints).any()
                if person_quality:
                    segment_data[person_id][idx] = 1
                else:
                    segment_data[person_id][idx] = -1
        data[seg_name] = segment_data
    return data

def plot_frame_quality_per_person(data: Dict[str, Dict[str, np.ndarray]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(["crimson", "lightgray", "forestgreen"])
    cmap.set_bad("white")

    for seg_name, persons in data.items():
        if not persons:
            continue
        person_ids = list(persons.keys())
        quality_matrix = np.array([persons[pid] for pid in person_ids], dtype=float)
        num_people, num_frames = quality_matrix.shape
        gap = 1 if num_people > 1 else 0
        rows_with_gaps = num_people * (gap + 1) - gap
        expanded = np.full((rows_with_gaps, num_frames), np.nan, dtype=float)
        for idx, row in enumerate(quality_matrix):
            expanded[idx * (gap + 1), :] = row
        masked = np.ma.masked_invalid(expanded)

        fig_height = max(3, num_people * 0.6)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        yticks = [i * (gap + 1) for i in range(num_people)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(person_ids)
        ax.set_title(f"Segment {seg_name} - Person Frame Quality")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Person ID")
        ax.set_xlim(0, num_frames)
        ax.set_ylim(rows_with_gaps - 0.5, -0.5)
        fig.tight_layout()

        fig_path = output_dir / f"{seg_name}_person_quality.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

def valid_person_mask(frequency: int=60) -> Dict[str, Dict[str, np.ndarray]]:
    frame_quality = pd.read_pickle("../data/export/intermediate/frame_quality.pkl")
    groups_annotations = extract_group_annotations(GROUP_ANNOTATIONS_PATH)

    # get annotation for each segment
    for seg_name, persons in frame_quality.items():
        cam, vid, seg = int(seg_name[0]), int(seg_name[1]), int(seg_name[2])
        # Annotations keyed by seg -> cam -> "mm:ss"
        annotations = groups_annotations.get(seg, {}).get(cam, {})
        if not annotations:
            continue

        person_ids = list(persons.keys())
        quality_all = np.vstack([persons[pid] for pid in person_ids])
        num_people, num_frames = quality_all.shape
        num_seconds = (num_frames + frequency - 1) // frequency

        # Convert per-second annotations into a frame-level mask.
        for sec_idx in range(num_seconds):
            start = sec_idx * frequency
            end = min(num_frames, start + frequency)
            # Offset each segment by 2 minutes; example: seg=3, sec=5 -> 04:05
            total_seconds = ((seg - 1) * 120) + sec_idx
            if vid == 3:  # video 3 starts from 00:01
                total_seconds += 1
            minutes, seconds = divmod(total_seconds, 60)
            ts_key = f"{minutes:02d}:{seconds:02d}"
            groups = annotations.get(ts_key)
            if not groups:
                continue

            allowed_people = set(pid for group in groups for pid in group)
            for p_idx, pid in enumerate(person_ids):
                if int(pid) not in allowed_people:
                    quality_all[p_idx, start:end] = 0  # mask out frames for unannotated person

        # Write masked arrays back into dict
        for p_idx, pid in enumerate(person_ids):
            persons[pid] = quality_all[p_idx]
    return frame_quality

def filter_people_dense() -> pd.DataFrame:
    all_kps = pd.read_pickle("../data/export/data_dense_all.pkl")
    frame_quality = valid_person_mask()
    all_kps['quality'] = 0
    for seg_name, persons in frame_quality.items():
        cam, vid, seg = int(seg_name[0]), int(seg_name[1]), int(seg_name[2])
        seg_name = f"{cam}{vid}{seg}"
        if not seg_name in SEGS:
            print(f"seg {seg_name} not in SEGS, skipping...")
            continue
        seg_kps_idx = (all_kps['Cam'] == cam) & (all_kps['Vid'] == vid) & (all_kps['Seg'] == seg)
        seg_rows = all_kps.loc[seg_kps_idx].sort_values("Timestamp")
        
        for person_id, quality in persons.items():
            
            if len(seg_rows) != len(quality):
                raise ValueError(f"Quality length mismatch for seg {seg_name}, person {person_id}: "
                                 f"{len(seg_rows)} rows vs {len(quality)} quality entries")

            # Iterate frame-by-frame; drop person key if quality != 1
            for frame_idx, row_idx in enumerate(seg_rows.index):
                if quality[frame_idx] != 1:
                    coords = all_kps.at[row_idx, "pixelCoords"]
                    if isinstance(coords, dict) and str(person_id) in coords:
                        new_coords = coords.copy()
                        new_coords.pop(str(person_id), None)
                        all_kps.at[row_idx, "pixelCoords"] = new_coords
                else:
                    all_kps.at[row_idx, "quality"] = 1

    return all_kps

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


    # raw_json_path = Path("/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco/")
    # write_path = Path("../data/results/segment_quality_dense/")
    # plot_all_segments_dense(json_parent=raw_json_path, output_dir=write_path)

    # frame_quality = frame_quality_person()
    # filehandler = open("../data/export/intermediate/frame_quality.pkl", 'wb')
    # pickle.dump(frame_quality, filehandler)
    
    # plot_frame_quality_per_person(frame_quality, Path("../data/results/segment_quality_person/"))
    filtered_kps = filter_people_dense()
    c = 9
