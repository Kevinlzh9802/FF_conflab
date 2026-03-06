from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Patch

CLUES: Tuple[str, ...] = ("head", "shoulder", "hip", "foot")
GroupState = Tuple[Tuple[int, ...], ...]


def _normalize_groups(groups: object) -> List[List[int]]:
    normalized: List[List[int]] = []
    if groups is None:
        return normalized
    try:
        items = list(groups)  # type: ignore[arg-type]
    except TypeError:
        items = [groups]

    for group in items:
        members: List[int] = []
        if isinstance(group, (list, tuple, set, np.ndarray)):
            for member in group:
                try:
                    members.append(int(member))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                members.append(int(group))
            except (TypeError, ValueError):
                members = []

        if members:
            normalized.append(sorted(set(members)))
    return normalized


def _extract_present_ids(row: pd.Series,
                         clue: str,
                         source_priority: Sequence[str]) -> set[int]:
    present: set[int] = set()
    for column in source_priority:
        value = row.get(column)
        if value is None:
            continue
        if not isinstance(value, dict) or clue not in value:
            continue
        feat = value.get(clue)
        if feat is None:
            continue
        arr = np.asarray(feat)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        for pid in arr[:, 0]:
            try:
                present.add(int(pid))
            except (TypeError, ValueError):
                continue
        if present:
            return present
    return present


def _parse_target_ids(target_ids: Sequence[int] | str) -> List[int]:
    if isinstance(target_ids, str):
        tokens = [token.strip() for token in target_ids.split(",") if token.strip()]
        return [int(token) for token in tokens]
    return [int(pid) for pid in target_ids]


def _build_target_state(row: pd.Series,
                        clue: str,
                        target_ids: Sequence[int],
                        target_set: set[int],
                        source_priority: Sequence[str]) -> GroupState:
    groups = _normalize_groups(row.get(f"{clue}Res"))
    selected_groups: List[Tuple[int, ...]] = []
    covered_targets: set[int] = set()
    for group in groups:
        group_set = set(group)
        if group_set & target_set:
            selected_groups.append(tuple(group))
            covered_targets.update(group_set & target_set)

    present_ids = _extract_present_ids(row=row, clue=clue, source_priority=source_priority)
    if not present_ids:
        # Fallback when features are unavailable: assume requested targets should be shown.
        present_ids = set(target_ids)
    for target_id in target_ids:
        if target_id in present_ids and target_id not in covered_targets:
            selected_groups.append((int(target_id),))

    unique_groups = sorted(set(selected_groups))
    return tuple(unique_groups)


def _state_label(state: GroupState) -> str:
    if not state:
        return "[]"
    return " | ".join("[" + ",".join(str(member) for member in group) + "]" for group in state)


def _state_color_map(states: Iterable[GroupState]) -> Dict[GroupState, Tuple[float, float, float]]:
    unique_states = sorted(set(states), key=lambda state: (len(state), state))
    if not unique_states:
        return {}

    group_counts = [len(state) for state in unique_states]
    min_groups = min(group_counts)
    max_groups = max(group_counts)
    group_span = max(1, max_groups - min_groups)

    color_map: Dict[GroupState, Tuple[float, float, float]] = {}
    for state in unique_states:
        n_groups = len(state)
        ratio = (n_groups - min_groups) / group_span

        seed_hex = hashlib.md5(repr(state).encode("utf-8")).hexdigest()[:8]
        seed_val = int(seed_hex, 16) / float(0xFFFFFFFF)

        # Low group count: dark/cool. High group count: bright/warm.
        base_hue = 0.62 - (0.56 * ratio)
        hue = (base_hue + (seed_val - 0.5) * 0.08) % 1.0
        saturation = float(np.clip(0.45 + (0.35 * ratio) + (seed_val - 0.5) * 0.08, 0.35, 0.95))
        value = float(np.clip(0.35 + (0.6 * ratio), 0.25, 1.0))
        rgb = tuple(float(x) for x in hsv_to_rgb([hue, saturation, value]))
        color_map[state] = rgb

    return color_map


def plot_target_grouping_spectrum(data_kp: pd.DataFrame,
                                  target_ids: Sequence[int] | str,
                                  clues: Sequence[str] = CLUES,
                                  source_priority: Sequence[str] = ("spaceFeat", "pixelFeat"),
                                  figsize: Tuple[float, float] = (16.0, 5.5),
                                  save_path: Optional[Path] = None,
                                  show: bool = True):
    """
    Plot grouping spectra over time for selected people IDs across all clues.

    Args:
        data_kp: DataFrame containing clue result columns, e.g., headRes/shoulderRes/hipRes/footRes.
        target_ids: Person IDs to track, e.g., [1, 2, 3, 4].
        clues: Clues to draw as rows (default: head, shoulder, hip, foot).
        source_priority: Feature columns used to infer present IDs for singleton reconstruction.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        show: Whether to display with plt.show().

    Returns:
        fig, ax, states_by_clue, color_map
    """
    parsed_target_ids = _parse_target_ids(target_ids)
    if len(parsed_target_ids) == 0:
        raise ValueError("target_ids cannot be empty.")

    if not isinstance(data_kp, pd.DataFrame):
        data_kp = pd.DataFrame(data_kp)

    target_ids_ordered = [int(pid) for pid in dict.fromkeys(parsed_target_ids)]
    target_set = set(target_ids_ordered)
    if len(data_kp) == 0:
        raise ValueError("data_kp is empty.")

    valid_clues = [str(clue) for clue in clues if f"{clue}Res" in data_kp.columns]
    if not valid_clues:
        raise ValueError(f"No clue result columns found for clues={list(clues)}.")

    states_by_clue: Dict[str, List[GroupState]] = {clue: [] for clue in valid_clues}
    all_states: List[GroupState] = []

    for frame_idx in range(len(data_kp)):
        row = data_kp.iloc[frame_idx]
        for clue in valid_clues:
            state = _build_target_state(
                row=row,
                clue=clue,
                target_ids=target_ids_ordered,
                target_set=target_set,
                source_priority=source_priority,
            )
            states_by_clue[clue].append(state)
            all_states.append(state)

    color_map = _state_color_map(all_states)
    color_image = np.zeros((len(valid_clues), len(data_kp), 3), dtype=float)
    for row_idx, clue in enumerate(valid_clues):
        for frame_idx, state in enumerate(states_by_clue[clue]):
            color_image[row_idx, frame_idx, :] = color_map[state]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(color_image, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(valid_clues)))
    ax.set_yticklabels([clue.capitalize() for clue in valid_clues])
    ax.set_xlabel("Frame Index")
    ax.set_title(f"Grouping Spectrum for IDs {target_ids_ordered}")

    legend_states = sorted(set(all_states), key=lambda state: (len(state), state))
    legend_handles = [
        Patch(facecolor=color_map[state], edgecolor="none",
              label=f"{_state_label(state)} ({len(state)} groups)")
        for state in legend_states
    ]

    if legend_handles:
        ncols = max(1, min(4, math.ceil(math.sqrt(len(legend_handles)))))
        nrows = math.ceil(len(legend_handles) / ncols)
        bottom_margin = min(0.6, 0.12 + 0.07 * nrows)
        fig.subplots_adjust(bottom=bottom_margin)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=ncols,
            frameon=False,
            fontsize=8,
            title="Grouping -> Color",
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, states_by_clue, color_map


# def compute_group_spectrum(data_kp: pd.DataFrame,
#                            target_ids: Sequence[int] | str,
#                            clues: Sequence[str] = CLUES,
#                            source_priority: Sequence[str] = ("spaceFeat", "pixelFeat"),
#                            figsize: Tuple[float, float] = (16.0, 5.5),
#                            save_path: Optional[Path] = None,
#                            show: bool = True):
#     """Backward-compatible alias for plot_target_grouping_spectrum."""
#     return plot_target_grouping_spectrum(
#         data_kp=data_kp,
#         target_ids=target_ids,
#         clues=clues,
#         source_priority=source_priority,
#         figsize=figsize,
#         save_path=save_path,
#         show=show,
#     )
