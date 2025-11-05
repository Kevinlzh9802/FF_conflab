from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from analysis.spatial import spatial_scores_df
from utils.groups import equal_groups

FEATURE_COLS = ['headRes', 'shoulderRes', 'hipRes', 'footRes']

# Probably deprecated in the future
# def construct_formations(results: dict, data: pd.DataFrame, speaking_status: Dict[str, Any] | None = None):
#     """Simplified port of utils/constructFormations.m.

#     Builds a formations table from unique groups per Vid for the 'GT' column.
#     Adds cardinality, id, participants, timestamps (from concat_ts), Cam, Vid.
#     If speaking_status provided with merged arrays per vid, computes avg_speaker.
#     Returns a pandas DataFrame 'formations'.
#     """
#     from utils.groups import record_unique_groups

#     col_name = 'GT'
#     vids = sorted(pd.unique(data['Vid']))
#     rows: List[Dict[str, Any]] = []
#     for vid in vids:
#         ana = data.loc[data['Vid'] == vid]
#         uniq = record_unique_groups(ana, col_name)
#         for entry in uniq:
#             rows.append({
#                 'participants': entry['participants'],
#                 'timestamps': entry['timestamps'],
#                 'timestamps_all': entry['timestamps'],
#                 'Cam': entry['Cam'],
#                 'Vid': vid,
#             })
#     if not rows:
#         return pd.DataFrame(columns=['participants', 'timestamps', 'timestamps_all', 'Cam', 'Vid', 'cardinality', 'id', 'avg_speaker'])
#     formations = pd.DataFrame(rows)
#     formations['cardinality'] = formations['participants'].apply(lambda x: len(x))
#     formations['id'] = np.arange(1, len(formations) + 1)
#     # Filter basic constraints
#     formations = formations[formations['cardinality'] >= 1]
#     # Optional: compute avg_speaker if speaking_status provided as dict per vid: array with first row IDs, following rows status
#     def _avg_speaker(row):
#         if speaking_status is None:
#             return np.nan
#         vid = int(row['Vid'])
#         actions = speaking_status.get(vid)
#         if actions is None:
#             return np.nan
#         ids = actions[0, :]
#         ts = np.asarray(row['timestamps_all'], dtype=int)
#         cols = []
#         for p in row['participants']:
#             loc = np.where(ids == p)[0]
#             if loc.size:
#                 cols.append(int(loc[0]))
#         if not cols or ts.size == 0:
#             return np.nan
#         sp = actions[1:, :]
#         ts = np.clip(ts, 1, sp.shape[0])
#         vals = sp[ts - 1][:, cols]
#         return float(np.sum(vals) / max(vals.size, 1))

#     formations['avg_speaker'] = formations.apply(_avg_speaker, axis=1)
#     return formations


def detect_group_num_breakpoints(data: pd.DataFrame, clues: List[str] | None = None):
    """Port of utils/detectGroupNumBreakpoints.m (simplified).

    - Finds breakpoints where any clue's groups change for each (Vid, Cam)
    - Generates windows between consecutive breakpoints
    - Returns a DataFrame window_table with columns: id, Vid, Cam, time, length, speaking_all_time
    """
    if clues is None:
        clues = FEATURE_COLS
    videos = sorted(pd.unique(data['Vid']))
    cameras = sorted(pd.unique(data['Cam']))
    rows: List[Dict[str, Any]] = []
    for vid in videos:
        for cam in cameras:
            cam_data = data[(data['Vid'] == vid) & (data['Cam'] == cam)]
            if cam_data.empty or 'concat_ts' not in cam_data.columns:
                continue
            ts = sorted(pd.unique(cam_data['concat_ts'].dropna()))
            if len(ts) <= 1:
                continue
            breakpoints: List[int] = []
            prev_groups = None
            groups_by_timestamp: Dict[int, List[Any]] = {}
            for t in ts:
                current_groups = []
                for clue in clues:
                    row_idx = cam_data.index[cam_data['concat_ts'] == t]
                    if len(row_idx) > 0 and clue in cam_data.columns:
                        current_groups.append(cam_data.loc[row_idx[0], clue])
                    else:
                        current_groups.append([])
                groups_by_timestamp[int(t)] = current_groups
                if prev_groups is None or any(not equal_groups(a, b) for a, b in zip(current_groups, prev_groups)):
                    breakpoints.append(t)
                prev_groups = current_groups
            # ensure first and last
            if breakpoints and breakpoints[0] != ts[0]:
                breakpoints = [ts[0]] + breakpoints
            if breakpoints and breakpoints[-1] != ts[-1]:
                breakpoints = breakpoints + [ts[-1]]
            # windows
            for i in range(len(breakpoints) - 1):
                start = int(breakpoints[i])
                end = int(breakpoints[i + 1])
                window_groups = groups_by_timestamp.get(start, [[] for _ in clues])
                row_dict = {
                    'id': i + 1,
                    'Vid': int(vid),
                    'Cam': int(cam),
                    'time': [start, end],
                    'length': end - start + 1,
                    'speaking_all_time': [],
                }
                for clue_name, groups_val in zip(clues, window_groups):
                    row_dict[clue_name] = groups_val
                rows.append(row_dict)
    return pd.DataFrame(rows)
    
def groups_speaker_belongs_clues(row, speakers: List):
    group_nums = {}
    for clue in FEATURE_COLS:
        detection = row.get(clue, 0)
        group_nums[clue] = groups_speaker_belongs(detection, speakers)
    return group_nums


def groups_speaker_belongs(groups: List, speakers: List):
    if len(speakers) == 0:
        return []
    groups_contained = groups.copy()
    remove_list = []
    for g in groups_contained:
        if len(set(g).intersection(set(speakers))) == 0:
            remove_list.append(g)
    for g in remove_list:
        groups_contained.remove(g)
    return groups_contained


def count_speaker_groups(
        breakpoints: pd.DataFrame,
        speaking_status_path: Optional[str] = "../data/export/sp_merged.pkl", 
) -> pd.DataFrame:
    """Augment breakpoint windows with speaking metadata.

    Parameters
    ----------
    breakpoints:
        DataFrame produced by `detect_group_num_breakpoints` (or similar) with at
        least the columns `Vid`, `time` (start/end frame pair), and optional `Cam`.
    speaking_status_path:
        Path to a merged speaking-status pickle generated by `write_sp_merged`.
    group_counter:
        Optional callable `(speakers: Sequence[int], context: Dict) -> Any` used to
        derive per-window group membership information. The return value is stored in
        the `scene_groups` column, and its length/size (if applicable) populates
        `num_scene_groups`. When omitted, `scene_groups` is set to ``None`` and the
        count defaults to zero.

    Returns
    -------
    pandas.DataFrame
        Copy of ``breakpoints`` with the additional columns:
          - ``speaking_window``: list of IDs speaking throughout the window.
          - ``speaking_in_scene``: filtered “in scene” speakers.
          - ``num_speaking_in_scene``: count of in-scene speakers.
          - ``scene_groups``: result from ``group_counter`` (or ``None``).
          - ``num_scene_groups``: number of groups (derived from ``scene_groups``).
    """
    if not isinstance(breakpoints, pd.DataFrame):
        raise TypeError("breakpoints must be a pandas DataFrame.")

    try:
        with open(speaking_status_path, "rb") as fh:
            sp_payload = pickle.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Speaking-status file not found: {speaking_status_path}") from exc

    speaking: Dict[int, np.ndarray] = sp_payload.get("speaking", {}) if isinstance(sp_payload, dict) else {}
    df = breakpoints.copy()
    df = df.reset_index(drop=True)

    def _normalize_groups_cell(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, float) and np.isnan(value):
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return [value]

    groups_columns = ['headRes', 'shoulderRes', 'hipRes', 'footRes']
    for col in groups_columns:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        else:
            df[col] = df[col].apply(_normalize_groups_cell)

    speaking_col: List[List[int]] = []
    in_scene_col: List[List[int]] = []
    num_scene_col: List[int] = []
    groups_sp_belong_col: List[Any] = []
    num_groups_sp_belong_col: List[int] = []
    all_people_col: List[List[int]] = []

    for idx, row in df.iterrows():
        vid = int(row.get("Vid", 0))
        time_window = row.get("time", [0, 0])
        if not isinstance(time_window, (list, tuple)) or len(time_window) < 2:
            time_window = [0, 0]
        start = int(time_window[0])
        end = int(time_window[1])
        if end < start:
            start, end = end, start

        status_matrix = speaking.get(vid)
        speakers_in_window: List[int] = []
        if isinstance(status_matrix, np.ndarray) and status_matrix.ndim == 2 and status_matrix.shape[0] >= 2:
            ids = status_matrix[0, :].astype(int)
            speaking_data = status_matrix[1:, :]
            # Convert to zero-based indices; ensure bounds.
            start_idx = max(0, start - 1)
            end_idx = min(speaking_data.shape[0] - 1, end - 1)
            if end_idx >= start_idx:
                window_slice = speaking_data[start_idx:end_idx + 1, :]
                if window_slice.size:
                    mask = np.all(window_slice == 1, axis=0)
                    speakers_in_window = ids[mask].astype(int).tolist()

        speaking_col.append(speakers_in_window)

        group_detection = row.get('headRes')
        people_in_scene = sorted({pid for group in group_detection for pid in group})
        speakers_in_scene = [pid for pid in speakers_in_window if pid in people_in_scene]
        all_people_col.append(people_in_scene)


        speakers_in_scene = [pid for pid in speakers_in_window if pid in people_in_scene]
        in_scene_col.append(speakers_in_scene)
        num_scene = len(speakers_in_scene)
        num_scene_col.append(num_scene)

        groups_speaker_belong = groups_speaker_belongs_clues(row, speakers_in_scene)
        num_groups_belong = {clue:len(g) for clue, g in groups_speaker_belong.items()}

        groups_sp_belong_col.append(groups_speaker_belong)
        num_groups_sp_belong_col.append(num_groups_belong)

    df["all_people_in_scene"] = all_people_col
    df["speaking"] = speaking_col
    df["speaking_in_scene"] = in_scene_col
    df["num_speaking_in_scene"] = num_scene_col
    df["groups_speaker_belong"] = groups_sp_belong_col
    df["num_groups_speaker_belong"] = num_groups_sp_belong_col

    return df


def filter_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Apply post-processing filters to the GCFF window table."""
    if windows is None:
        raise ValueError("windows dataframe must not be None")

    result_cols = FEATURE_COLS
    if windows.empty:
        return windows.copy()

    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, (pd.Series, pd.DataFrame)) and value.empty:
            return True
        try:
            if len(value) == 0:  # type: ignore[arg-type]
                return True
        except TypeError:
            pass
        return False

    def _normalize_groups(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return value.values.tolist()
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    def _all_identical(row: pd.Series) -> bool:
        values = [_normalize_groups(row[col]) for col in result_cols]
        if not values:
            return True
        first = values[0]
        for other in values[1:]:
            if not equal_groups(first, other):
                return False
        return True

    non_empty_mask = ~windows[result_cols].applymap(_is_empty).any(axis=1)
    identical_mask = windows[result_cols].apply(_all_identical, axis=1)
    speaking_mask = windows['num_speaking_in_scene'] > 1

    keep_mask = non_empty_mask & ~identical_mask & speaking_mask
    return windows.loc[keep_mask].copy()

def speaker_group_diff_plot(windows: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> None:
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if windows.empty:
        return

    diff_df = pd.DataFrame({
        feature: windows['num_speaking_in_scene'] - windows['num_groups_speaker_belong'].apply(
            lambda entry: (entry if isinstance(entry, dict) else {}).get(feature, 0)
        ) for feature in feature_cols
    })

    diff_values = sorted({
        int(val)
        for val in diff_df.to_numpy().ravel()
        if pd.notna(val)
    })
    if not diff_values:
        return

    x = np.arange(len(diff_values))
    width = 0.2
    offsets = np.linspace(-width * (len(feature_cols) - 1) / 2,
                          width * (len(feature_cols) - 1) / 2,
                          len(feature_cols))
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, feature in zip(offsets, feature_cols):
        counts = diff_df[feature].value_counts().reindex(diff_values, fill_value=0)
        ax.bar(x + offset, counts.values, width=width, label=feature)

    ax.set_xlabel('num_groups_speaker_belong - num_speaking_in_scene')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in diff_values])
    ax.legend(title='Feature set')
    ax.set_title('Distribution of group memberships minus in-scene speakers')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plt.show()


def speaker_group_bubble_plot(windows: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> None:
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if windows.empty:
        return

    colors = plt.cm.get_cmap('tab10', len(feature_cols))
    overall_points: List[Tuple[int, int]] = []
    width = 0.15
    offsets = np.linspace(
        -width * (len(feature_cols) - 1) / 2,
        width * (len(feature_cols) - 1) / 2,
        len(feature_cols),
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (feature, offset) in enumerate(zip(feature_cols, offsets)):
        data_points: List[Tuple[int, int]] = []
        for _, row in windows.iterrows():
            detection_result = row.get(feature, [])
            if detection_result is None:
                continue
            filtered_speakers = row.get('speaking_in_scene', []) or []

            for group in detection_result:
                if group is None:
                    continue
                if isinstance(group, np.ndarray):
                    members = group.tolist()
                elif isinstance(group, (list, tuple)):
                    members = list(group)
                else:
                    continue

                group_size = len(members)
                if group_size <= 1:
                    continue

                filtered_count = sum(member in filtered_speakers for member in members)
                data_points.append((group_size, filtered_count))

        if not data_points:
            continue

        counts = Counter(data_points)
        unique_combinations = np.array(list(counts.keys()))
        if unique_combinations.ndim == 1:
            unique_combinations = unique_combinations.reshape(1, -1)

        sizes = np.array(list(counts.values()))
        ax.scatter(
            unique_combinations[:, 0] + offset,
            unique_combinations[:, 1],
            s=sizes * 60,
            c=[colors(idx)],
            label=feature,
            alpha=0.7,
            edgecolors='k',
            linewidths=1.5,
        )
        overall_points.extend(data_points)

    if not overall_points:
        plt.close(fig)
        return

    ax.set_xlabel('Detected Group Size')
    ax.set_ylabel('Number of Simultaneous Speakers in Group')
    ax.set_title('Group Size vs. Speakers in Scene by Feature')
    ax.legend(title='Feature set')
    ax.grid(axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

    group_sizes = [pt[0] for pt in overall_points]
    speaker_counts = [pt[1] for pt in overall_points]
    max_group_size = max(group_sizes, default=2)
    max_speakers = max(speaker_counts, default=1)
    ax.set_xlim(1, max_group_size + 1)
    ax.set_ylim(-0.5, max_speakers + 1)

    fig.tight_layout()
    plt.show()


def cross_modal_analysis(data):
    windows = detect_group_num_breakpoints(data=data)
    windows = count_speaker_groups(windows)
    windows = filter_windows(windows)
    
    # speaker_group_diff_plot(windows, FEATURE_COLS)
    # speaker_group_bubble_plot(windows, FEATURE_COLS)
    spatial_scores_df(windows, FEATURE_COLS)
    return windows
