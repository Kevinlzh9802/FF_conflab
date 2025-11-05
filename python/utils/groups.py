from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple
import itertools


def unique_cell_groups(cell_array: Sequence[Sequence[Sequence[int]]]) -> Tuple[List[List[List[int]]], List[int]]:
    """Deduplicate a cell array of groups of IDs.

    Each entry is a list of groups (each group is list[int]). Two entries are
    considered equal if all groups match up to element ordering.
    Returns (unique_values, counts).
    """
    def norm(entry: Sequence[Sequence[int]]) -> str:
        parts = [" ".join(map(str, sorted(g))) for g in entry]
        return "|".join(parts)

    keys = [norm(e) for e in cell_array]
    uniq = {}
    for k, e in zip(keys, cell_array):
        if k in uniq:
            uniq[k][1] += 1
        else:
            uniq[k] = [list(map(list, e)), 1]
    unique_vals = [v[0] for v in uniq.values()]
    counts = [v[1] for v in uniq.values()]
    return unique_vals, counts


def set_coverage(A: List[List[int]], B: List[List[int]]) -> List[int]:
    """Compute coverage scores between two partitions A and B.

    Returns [a_count, b_count] as in MATLAB setCoverage.m helper logic.
    """
    def cover_side(X: List[List[int]], Y: List[List[int]]) -> int:
        total = 0
        for target in Y:
            total += find_group_cover_score(target, X)
        return total

    return [cover_side(A, B), cover_side(B, A)]


def find_group_cover_score(target: List[int], group_set: List[List[int]]) -> int:
    """Score for how well a target set is covered by unions of sets in group_set.

    - 0 if target is equal to or subset of any group in group_set
    - 1 if union of some combination equals target
    - else 0 (keeping definition simple and bounded like MATLAB's early exits)
    """
    tset = set(target)
    for g in group_set:
        gset = set(g)
        if tset.issubset(gset) or tset == gset:
            return 0
    n = len(group_set)
    for k in range(2, n + 1):
        for combo in itertools.combinations(group_set, k):
            u = set().union(*map(set, combo))
            if u == tset:
                return 1
    return 0


def filter_group_by_members(vector: Iterable[int], groups: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return groups that share at least one member with vector."""
    s = set(vector)
    out: List[List[int]] = []
    for g in groups:
        if any((x in s) for x in g):
            out.append(list(g))
    return out


def record_unique_groups(T, g_name: str):
    """Collect unique groups in a table-like object T with columns g_name, concat_ts, Cam.

    Returns a list of dicts with participants (list[int]), timestamps (list[int/float]), Cam (unique cams).
    """
    group_map = {}
    for k in range(len(T)):
        current_groups = T[g_name][k]
        current_time = T['concat_ts'][k]
        current_cam = T['Cam'][k]
        if not current_groups:
            continue
        for grp in current_groups:
            key = " ".join(map(str, sorted(grp)))
            entry = {'timestamp': current_time, 'cam': current_cam}
            group_map.setdefault(key, []).append(entry)
    result = []
    for key, entries in group_map.items():
        participants = list(map(int, key.split())) if key else []
        timestamps = [e['timestamp'] for e in entries]
        cams = sorted(set(e['cam'] for e in entries))
        result.append({'participants': participants, 'timestamps': timestamps, 'Cam': cams})
    return result


def turn_singletons_to_groups(groups: List) -> List[List[int]]:
    groups_copy = groups.copy()
    for idx in range(len(groups)):
        if isinstance(groups[idx], int):
            groups_copy[idx] = [groups[idx]]
    return groups_copy


def equal_groups(a, b):
    try:
        if a is None and b is None:
            return True
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            norm_a = sorted(sorted(x) for x in a)
            norm_b = sorted(sorted(x) for x in b)
            return norm_a == norm_b
        return a == b
    except Exception:
        return False