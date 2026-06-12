"""
Python port of GCFF/example_GCFF.m as an executable module.

This script orchestrates the GCFF pipeline over a sequence of frames.
It expects pre-loaded data structures and calls into Python ports of the
original MATLAB functions. Script-style .m files are represented here as
function calls with parameters (see utils.python.scripts stubs).

Typical usage (pseudo-code):
    from GCFF.python.main_GCFF import run_gcff_sequence, Params
    results, data_out = run_gcff_sequence(data, Params(stride=40, mdl=6000),
                                          clue="head", speaking_status=sp, use_real=True)

Data expectations:
    data: a dict-like or pandas.DataFrame with columns:
        - headFeat, shoulderFeat, hipFeat, footFeat (per-frame arrays)
        - GT (per-frame list of groups) [optional]
        - Cam, Vid, Seg, Timestamp (scalars per frame)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import os
import sys
import numpy as np
import argparse
import pandas as pd
import pickle
import yaml
from munch import Munch, unmunchify
from datetime import datetime

from gcff_core import ff_deletesingletons, ff_evalgroups, graph_cut
from analysis.cross_modal import cross_modal_analysis
from utils.table import filter_and_concat_table
from utils.groups import turn_singletons_to_groups
from utils.plots import plot_all_skeletons, plot_panels_df
from utils.exp_logging import display_frame_results, start_logging, stop_logging, log_only, get_log_path
# from utils.ground_csv_to_data_kp import convert_ground_csv_to_data_kp
from tests.data_quality import annotate_frame_quality
from tests.group_spectrum import plot_target_grouping_spectrum


def _resolve_paths(config: Munch) -> tuple[str, str]:
    """Return (input_path, finished_path) from config.paths.kp / kp_finished."""
    return str(config.paths.kp), str(config.paths.kp_finished)


def _collect_all_person_ids(batch_df: pd.DataFrame) -> List[int]:
    """Return sorted list of unique person IDs from spaceFeat across all clues and rows."""
    all_ids: set = set()
    for _, row in batch_df.iterrows():
        sf = row.get("spaceFeat") or {}
        if not isinstance(sf, dict):
            continue
        for arr in sf.values():
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 1:
                for pid in a[:, 0]:
                    try:
                        all_ids.add(int(pid))
                    except (ValueError, TypeError):
                        pass
    return sorted(all_ids)


def _plot_spectrum_per_batch(data_kp: pd.DataFrame, spectrum_dir: Path) -> None:
    """Save one grouping-spectrum PNG per batch, using all detected person IDs."""
    import matplotlib.pyplot as plt
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    for (cam, vid, seg), batch_df in data_kp.groupby(["Cam", "Vid", "Seg"]):
        batch_num = f"{int(cam)}{int(vid)}{int(seg)}"
        save_path = spectrum_dir / f"{batch_num}.png"
        target_ids = _collect_all_person_ids(batch_df)
        if not target_ids:
            print(f"  Spectrum [{batch_num}]: no person IDs found, skipping.")
            continue
        try:
            plot_target_grouping_spectrum(
                data_kp=batch_df.reset_index(drop=True),
                target_ids=target_ids,
                save_path=save_path,
                show=False,
            )
            plt.close("all")
            print(f"  Spectrum [{batch_num}]: saved {save_path}")
        except Exception as exc:
            print(f"  Spectrum [{batch_num}]: failed: {exc}")


def gcff_experiments(config: Munch,
                     clue: Optional[str] = None,
                     detection_dir: Optional[Path] = None) -> pd.DataFrame:
    """Run GCFF detection.

    Per-clue mode: pass clue + detection_dir.  Saves a minimal detection pkl
    (.../detection/<clue>.pkl) and skips writing data_finished.pkl and plots.
    All-clues mode: clue=None, detection_dir=None — original behaviour.
    """
    kp_input, kp_finished = _resolve_paths(config)
    per_clue_mode = detection_dir is not None

    # Per-clue mode always reloads from data.pkl; all-clues mode respects force_rerun.
    if per_clue_mode or config.force_rerun:
        data_kp = pd.read_pickle(kp_input)
        rerun = True
    else:
        try:
            data_kp = pd.read_pickle(kp_finished)
            rerun = False
        except Exception:
            data_kp = pd.read_pickle(kp_input)
            rerun = True

    # filter and concat table by 3-digit keys in params.used_parts
    data_kp = filter_and_concat_table(data_kp, config.used_segs)

    # Test mode: keep a sparse subset so gcff_sequence runs faster.
    if config.test:
        data_kp = data_kp.iloc[::100].reset_index(drop=True)
        print(f"Test mode enabled: using every 100th row ({len(data_kp)} rows).")

    print(f"GCFF dataframe rows in use: {len(data_kp)}")

    if rerun:
        clues_to_run = [clue] if clue else list(config.all_clues)
        if per_clue_mode:
            detection_dir.mkdir(parents=True, exist_ok=True)

        for c in clues_to_run:
            print(f"\nRunning GCFF for clue: {c}")
            if config.use_space:
                features = [data_kp["spaceFeat"][k][c] for k in range(len(data_kp))]
            else:
                features = [data_kp["pixelFeat"][k][c] for k in range(len(data_kp))]
            GTgroups = list(data_kp['GT']) if ('GT' in getattr(data_kp, 'columns', [])) else [None] * len(features)
            frame_contexts = [_build_frame_debug_context(data_kp, k, clue=c) for k in range(len(data_kp))]

            results = gcff_sequence(features, GTgroups, config.params, frame_contexts=frame_contexts)
            data_kp[f"{c}Res"] = results['groups']

            if per_clue_mode:
                det_df = pd.DataFrame({
                    "Cam": data_kp["Cam"],
                    "Vid": data_kp["Vid"],
                    "Seg": data_kp["Seg"],
                    "Timestamp": data_kp["Timestamp"],
                    f"{c}Res": results['groups'],
                })
                det_path = detection_dir / f"{c}.pkl"
                det_df.to_pickle(det_path)
                print(f"  [{c}] Detection saved: {det_path}  ({len(det_df)} rows)")

        # All-clues mode: save data_finished.pkl and plots as before.
        if not per_clue_mode:
            if config.replace_df:
                data_kp.to_pickle(kp_finished)

            if config.plots.panels:
                _use_bev = (
                    "spaceCoords" not in data_kp.columns
                    or data_kp["spaceCoords"].iloc[0] is None
                )
                if _use_bev:
                    from utils.plot_spacefeat import plot_spacefeat_bev_panels_df
                    plot_spacefeat_bev_panels_df(data_kp, output_dir=config.paths.panel_plots)
                else:
                    plot_panels_df(data_kp, output_dir=config.paths.panel_plots)

                _spectrum_dir = Path(
                    getattr(config.paths, "spectrum_plots",
                            str(Path(config.paths.panel_plots).parent / "spectrum"))
                )
                _plot_spectrum_per_batch(data_kp, _spectrum_dir)

    return data_kp

def _build_frame_debug_context(data_kp: pd.DataFrame,
                               frame_idx: int,
                               clue: Optional[str] = None) -> Dict[str, Any]:
    context: Dict[str, Any] = {"frame_idx": int(frame_idx)}
    if clue is not None:
        context["clue"] = clue
    for key in ("row_id", "Cam", "Vid", "Seg", "Timestamp"):
        try:
            value = data_kp[key].iloc[frame_idx]
        except Exception:
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        context[key] = value
    return context


def _log_gcff_progress(processed_count: int,
                       total_count: int,
                       context: Optional[Dict[str, Any]] = None) -> None:
    context = context or {}
    clue = context.get("clue", "unknown")
    cam = context.get("Cam", "?")
    vid = context.get("Vid", "?")
    seg = context.get("Seg", "?")
    timestamp = context.get("Timestamp", "?")
    frame_idx = context.get("frame_idx", "?")
    print(
        f"GCFF progress: {processed_count}/{total_count} "
        f"(clue={clue}, frame_idx={frame_idx}, Cam={cam}, Vid={vid}, Seg={seg}, Timestamp={timestamp})"
    )


def gcff_sequence(features, GTgroups, params, frame_contexts: Optional[List[Dict[str, Any]]] = None):
    """High-level pipeline adapted from example_GCFF.m.

    Returns (results_dict, data_out)
    """
    T = len(features)
    TP = np.zeros(T)
    FP = np.zeros(T)
    FN = np.zeros(T)
    precision = np.zeros(T)
    recall = np.zeros(T)
    groups_out: List[List[List[int]]] = [None] * T
    s_speaker: List[float] = []
    group_sizes: List[int] = []

    for idx in range(T):
        feat = features[idx]
        if feat is None or len(feat) == 0 or feat.shape[1] == 0:
            groups_out[idx] = []
            continue

        debug_context = frame_contexts[idx] if frame_contexts is not None and idx < len(frame_contexts) else None
        processed_count = idx + 1
        if processed_count == 1 or processed_count % 500 == 0 or processed_count == T:
            _log_gcff_progress(processed_count, T, debug_context)
        labels = graph_cut(feat, params.stride, params.mdl, debug_context=debug_context)
        groups = []
        for lab in range(int(labels.max()) + 1 if labels.size else 0):
            members = feat[labels == lab, 0].astype(int).tolist()
            groups.append(members)

        # Deal with singletons
        if not ff_deletesingletons(groups):  # which means groups are all singletons
            groups = []
        groups = turn_singletons_to_groups(groups)
        # GT = turn_singletons_to_groups(GTgroups[idx])
        groups_out[idx] = groups

        # Evaluate
        # pr, re, tp, fp, fn = ff_evalgroups(groups, GT, TH='card', cardmode=0)
        # precision[idx], recall[idx], TP[idx], FP[idx], FN[idx] = pr, re, tp, fp, fn

        # display_frame_results(idx + 1, T, groups, GT)

    # pr_avg = float(np.nanmean(precision)) if precision.size else float('nan')
    # re_avg = float(np.nanmean(recall)) if recall.size else float('nan')
    # F1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
    # F1_avg = float(np.nanmean(F1)) if F1.size else float('nan')

    results = {
        'precisions': precision,
        'recalls': recall,
        'F1s': 0,
        'F1_avg': 0,
        'groups': groups_out,
        'group_sizes': np.array(group_sizes),
        's_speaker': np.array(s_speaker),
    }

    return results

def set_config(args):
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    
    config = Munch(config)
    for key, item in config.items():
        if isinstance(item, dict):
            config[key] = Munch(item)
    
    if args.stride is not None:
        config.params.stride = args.stride
    if args.mdl is not None:
        config.params.mdl = args.mdl
    if args.use_space is not None:
        config.use_space = args.use_space
    if args.force_rerun is not None:
        config.force_rerun = args.force_rerun
    config.test = bool(args.test)

    return config

if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Run GCFF main pipeline.')
    parser.add_argument('--config', type=str, default="./configs/config_GCFF.yaml", help='Path to config YAML file')
    parser.add_argument('--stride', type=float, default=None)
    parser.add_argument('--mdl', type=float, default=None)
    parser.add_argument('--use-space', type=bool, default=None)
    parser.add_argument('--force-rerun', type=bool, default=None)
    parser.add_argument('-test', '--test', action='store_true',
                        help='If set, sample every 100th row in data_kp before running gcff_sequence')
    parser.add_argument('--clue', default=None,
                        choices=["head", "shoulder", "hip", "foot"],
                        help="Single body clue to process. Omit to run all clues (default).")
    parser.add_argument('--detection-dir', default=None,
                        help="Directory for per-clue detection pkls. "
                             "Providing this enables per-clue mode (skips data_finished.pkl and plots).")

    args = parser.parse_args()
    config = set_config(args)

    detection_dir = Path(args.detection_dir) if args.detection_dir else None

    ts_display, log_path = get_log_path(config)
    start_logging(log_path)
    try:
        print(f"Logging to: {log_path}")
        log_only(f"Start time: {ts_display}")
        try:
            cfg_yaml = yaml.safe_dump(unmunchify(config), sort_keys=False, allow_unicode=True)
        except Exception:
            cfg_yaml = str(config)
        log_only("Config:\n\n" + cfg_yaml.rstrip())
        res = gcff_experiments(config, clue=args.clue, detection_dir=detection_dir)
    finally:
        end_dt = datetime.now()
        log_only(f"End time: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        stop_logging()
