import os
import pandas as pd
import numpy as np
import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

from utils.pose import process_foot_data, extract_raw_keypoints, construct_space_coords, process_orient


ALL_CLUES = ["head", "shoulder", "hip", "foot"]
RAW_jSON_PATH = Path("/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco/")
SEGS = ["429", "431", "631", "634", "636", "831", "832", "833", "833", "834", "835"]

def load_cam_params(cam: Union[int, str],
                    params_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load camera intrinsic/extrinsic parameters and metadata for the given camera id.

    Args:
        cam: Camera identifier (e.g., ``2`` or ``"2"``) matching the JSON filenames.
        params_dir: Optional override for the directory containing the JSON files.

    Returns:
        Dictionary containing intrinsic matrix ``K``, distortion coefficients, rotation ``R``,
        translation ``t``, plus height ratio and part-column maps.
    """
    cam_str = str(cam)
    if params_dir is None:
        params_dir = Path(__file__).resolve().parents[2] / "data" / "camera_params"
    base_path = Path(params_dir)
    intrinsic_path = base_path / f"intrinsic_{cam_str}.json"
    extrinsic_path = base_path / f"extrinsic_{cam_str}_zh.json"

    with open(intrinsic_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    with open(extrinsic_path, "r", encoding="utf-8") as f:
        extrinsics = json.load(f)

    height_ratios_map = {
        "head": 1.0,
        "nose": 0.95,
        "leftShoulder": 0.85,
        "rightShoulder": 0.85,
        "leftHip": 0.5,
        "rightHip": 0.5,
        "leftAnkle": 0.02,
        "rightAnkle": 0.02,
        "leftFoot": 0.02,
        "rightFoot": 0.02,
    }
    # part_column_map = {
    #     "head": (5, 6),
    #     "nose": (7, 8),
    #     "leftShoulder": (9, 10),
    #     "rightShoulder": (11, 12),
    #     "leftHip": (13, 14),
    #     "rightHip": (15, 16),
    #     "leftAnkle": (17, 18),
    #     "rightAnkle": (19, 20),
    #     "leftFoot": (21, 22),
    #     "rightFoot": (23, 24),
    # }

    return {
        "K": np.asarray(intrinsics.get("intrinsic"), dtype=float),
        "distCoeff": np.asarray(intrinsics.get("distortion_coefficients"), dtype=float),
        "R": np.asarray(extrinsics.get("rotation"), dtype=float),
        "t": np.asarray(extrinsics.get("translation"), dtype=float),
        "height_ratios_map": height_ratios_map,
        # "part_column_map": part_column_map,
        "bodyHeight": 170,
        "img_size": (1920, 1080),
    }

def process_data(base_dir):
    data = load_all_data(base_dir)

    # Process foot features if present
    if hasattr(data, 'columns') and ('footFeat' in data.columns):
        data = process_foot_data(data)

    # Translate script calls to functions (placeholders/stubs in utils)
    data = concat_segs(data)

    # Assign incremental ids if not present
    if hasattr(data, 'assign'):
        try:
            data = data.assign(id=np.arange(1, len(data) + 1))
        except Exception:
            pass
    data = process_columns(data)
    data.to_pickle(base_dir + "data.pkl")


def load_all_data(base_dir: str) -> pd.DataFrame:
    all_data = None
    for clue in ALL_CLUES:
        clue_dir = base_dir + clue
        data = load_data(clue_dir)
        if all_data is None:
            all_data = data
            all_data.rename(columns={"Features": clue+"Feat"}, inplace=True)
        else:
            all_data[clue+"Feat"] = data["Features"]
    return all_data

def load_data(base: str) -> pd.DataFrame:
    # 1) metadata
    meta = pd.read_csv(os.path.join(base, "metadata.csv"))

    # 2) GT
    gt_rows = []
    with open(os.path.join(base, "gt.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            gt_rows.append(json.loads(s))
    gt = pd.DataFrame(gt_rows)  # columns: row_id, GT (list-of-lists)

    # 3) lazy loader for features: path only, read on demand
    features = load_features(os.path.join(base, "features"), len(meta))

    # 4) Merge all together
    meta["Features"] = features
    meta = meta.merge(gt, on="row_id", how="left")

    return meta

def load_features(base_dir, nrows):
    feats = []
    for i in range(1, nrows + 1):
        fname = os.path.join(base_dir, f"f_{i:06d}.csv")
        if not os.path.exists(fname):
            feats.append(None)
            continue
        arr = np.loadtxt(fname, delimiter=",")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        feats.append(arr)
    return feats

def _add_merged_column(df: pd.DataFrame) -> pd.DataFrame: 
    ids = pd.unique(df['Seg'])
    offset = 0.0
    concat_ts = np.full((len(df),), np.nan, dtype=float)
    for seg_id in ids:
        idx = df['Seg'] == seg_id
        frames = df.loc[idx, 'Timestamp'].to_numpy(dtype=float)
        merged = offset + frames
        concat_ts[idx.to_numpy()] = merged
        offset = merged[-1]
    out = df.copy()
    out['concat_ts'] = concat_ts
    return out

def concat_segs(data: pd.DataFrame) -> pd.DataFrame:
    """Port of utils/concatSegs.m.

    For each (Vid, Cam), treat rows as a continuous time sequence and
    concatenate their timestamps based on the order of Seg values.

    Adds a new column 'concat_ts' to the DataFrame without altering
    original 'Seg' or 'Timestamp' values.
    """
    df = data.copy()
    df['concat_ts'] = np.nan
    if 'Vid' not in df.columns or 'Cam' not in df.columns:
        return df
    groups = df.groupby(['Vid', 'Cam'], sort=False)
    for (vid, cam), idx in groups.groups.items():
        sub = df.loc[idx, ['Seg', 'Timestamp']].copy()
        merged = _add_merged_column(sub)
        df.loc[idx, 'concat_ts'] = merged['concat_ts'].values
    return df

def process_columns(df):
    pixel_feats = []
    space_feats = []
    pixel_coords = []
    space_coords = []
    
    for _, row in df.iterrows():
        # store features from all clues
        pixel_feat = {clue: [] for clue in ALL_CLUES}
        space_feat = {clue: [] for clue in ALL_CLUES}
        for clue in ALL_CLUES:
            feat = row.get(f"{clue}Feat", None)
            if feat is not None and feat.shape[1] > 0:
                pixel_feat[clue] = feat[:, 0:4]
                space_feat[clue] = feat[:, 24:28]
            else:
                pixel_feat[clue] = np.array([])
                space_feat[clue] = np.array([])
        pixel_feats.append(pixel_feat)
        space_feats.append(space_feat)

        # Store coordinates from headFeat
        headFeat = row.get(f"headFeat", None)
        if headFeat is not None and headFeat.shape[1] > 0:
            pixel_coord = headFeat[:, 4:24]
            space_coord = headFeat[:, 28:48]
        else:
            pixel_coord = np.array([])
            space_coord = np.array([])
        pixel_coords.append(pixel_coord)
        space_coords.append(space_coord)

    df['pixelCoords'] = pixel_coords
    df['spaceCoords'] = space_coords   
    df["pixelFeat"] = pixel_feats
    df["spaceFeat"] = space_feats

    df.drop(columns=[f"{clue}Feat" for clue in ALL_CLUES], inplace=True)
    df = df['row_id Cam Vid Seg Timestamp concat_ts pixelCoords spaceCoords pixelFeat spaceFeat GT'.split()]
    return df

def save_group_to_csv(T, path: str = 'f_formations.csv') -> None:
    """Save a table with column 'formations' (list of groups) to CSV."""
    import csv

    rows = []
    for row in T:
        groups = row.get('formations') or []
        for g in groups:
            rows.append({'participants': " ".join(map(str, g))})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['participants'])
        w.writeheader()
        w.writerows(rows)

def read_dense_df(filter_segs: bool) -> pd.DataFrame:
    """
    Parse raw pose JSON files and produce a frame-level DataFrame.

    Each frame in every skeleton JSON is represented as a row with the columns
    listed in the comment at the bottom of this module. Cam, Vid, and Seg are
    inferred from the JSON filename pattern.
    """
    records: List[Dict[str, Any]] = []
    row_id = 1
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
        if filter_segs and (not seg_name in SEGS):
            continue

        with open(json_path, "r") as file:
            raw_annotation = json.load(file)
        skeletons = raw_annotation.get("annotations", {}).get("skeletons", [])
        camera_params = load_cam_params(cam=cam)

        for idx in range(len(skeletons)):
            frame_coords = extract_raw_keypoints(skeletons, idx)
            space_coords = construct_space_coords(frame_coords, camera_params)
            records.append(
                {
                    "row_id": row_id,
                    "Cam": cam,
                    "Vid": vid,
                    "Seg": seg,
                    "Timestamp": idx,
                    "concat_ts": idx,
                    "pixelCoords": frame_coords,
                    "spaceCoords": space_coords,
                    "pixelFeat": None,
                    "spaceFeat": None,
                    "headRes": None,
                    "shoulderRes": None,
                    "hipRes": None,
                    "footRes": None,
                }
            )
            row_id += 1

    columns = [
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
        "headRes",
        "shoulderRes",
        "hipRes",
        "footRes",
    ]
    data_kp = pd.DataFrame.from_records(records, columns=columns)
    
    return data_kp

def generate_dense_feats(data: pd.DataFrame) -> pd.DataFrame:
    for _, row in data.iterrows():
        person_ids, pixel_coords, space_coords = [], [], []
        pixel_info = row.get("pixelCoords", None)
        space_info = row.get("spaceCoords", None)
        for person_id in pixel_info.keys():
            person_ids.append(int(person_id))
            pixel_coords.append(pixel_info[person_id])
            space_coords.append(space_info[person_id])
        pixel_coords = np.stack(pixel_coords)
        space_coords = np.stack(space_coords)
        pixel_feat = process_orient(np.array(pixel_coords), [1920, 1080], True)
        space_feat = process_orient(np.array(space_coords), [1,1], False)

        row["pixelFeat"] = {clue: np.column_stack([person_ids, feat]) for clue, feat in pixel_feat.items()}
        row["spaceFeat"] = {clue: np.column_stack([person_ids, feat]) for clue, feat in space_feat.items()}
    return data


_GROUP_PATTERN = re.compile(r"\(<\s*([^>]+?)\s*>,\s*cam(\d+)\)", re.IGNORECASE)
_SEG_PATTERN = re.compile(r"seg(\d)\.csv", re.IGNORECASE)

def _sequential_timestamp(idx: int) -> str:
    """Format a zero-based row index as HH:MM, starting from 00:00 and stepping by one minute."""
    hours, minutes = divmod(idx, 60)
    return f"{hours:02d}:{minutes:02d}"

def _parse_group_cell(cell: str) -> Dict[int, List[List[int]]]:
    """Parse a single CSV cell and split it into per-camera group lists."""
    groups_by_cam: Dict[int, List[List[int]]] = defaultdict(list)
    if pd.isna(cell):
        return groups_by_cam

    for match in _GROUP_PATTERN.finditer(str(cell)):
        members_raw, cam_raw = match.groups()
        members = [int(p.strip()) for p in members_raw.split(",") if p.strip()]
        if members:
            groups_by_cam[int(cam_raw)].append(members)
    return groups_by_cam

def extract_group_annotations(data_path: Union[str, Path]) -> Dict[int, Dict[int, Dict[str, List[List[int]]]]]:
    """
    Extract group annotations keyed by video (seg), then camera, then timestamp.

    Timestamps are normalized to start at 00:00 for each seg file and increase
    sequentially by one minute per row.

    Returns a mapping of the form:
        {
            seg_id: {
                cam_id: {
                    "00:00": [[2, 3], [4, 5, 6]],
                    "00:01": [[...], ...],
                },
            },
            ...
        }
    """
    data_path = Path(data_path)
    seg_groups: Dict[int, Dict[int, Dict[str, List[List[int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for csv_path in sorted(data_path.glob("seg*.csv")):
        seg_match = _SEG_PATTERN.match(csv_path.name)
        if not seg_match:
            continue  # skip non-seg files and multi-digit seg ids
        seg_id = int(seg_match.group(1))

        df = pd.read_csv(csv_path, header=None, names=["timestamp", "groups"])
        for idx, row in df.iterrows():
            # timestamp = _sequential_timestamp(idx)
            parsed_groups = _parse_group_cell(row["groups"])
            for cam_id, groups in parsed_groups.items():
                seg_groups[seg_id][cam_id][row["timestamp"]].extend(groups)

    # Convert nested defaultdicts to plain dicts for safer downstream use.
    return {
        seg: {cam: dict(ts_groups) for cam, ts_groups in cam_map.items()}
        for seg, cam_map in seg_groups.items()
    }

def filter_people_dense(df):
    group_annotaion_path = Path("/home/zonghuan/tudelft/projects/datasets/conflab/annotations/f_formations")
    group_annotations = extract_group_annotations(group_annotaion_path)
    pass

# Index(['row_id', 'Cam', 'Vid', 'Seg', 'Timestamp', 'concat_ts', 'pixelCoords',
#        'spaceCoords', 'pixelFeat', 'spaceFeat', 'GT', 'headRes', 'shoulderRes',
#        'hipRes', 'footRes'],
#       dtype='object')
if __name__ == '__main__': 
    # process_data("../data/export/")
    df = read_dense_df(filter_segs=False)

    df.to_pickle("../data/export/data_dense_all.pkl")
    # df = pd.read_pickle("../data/export/data_dense.pkl")
    df = generate_dense_feats(df)
    df = filter_people_dense(df)

    # extract_group_annotations(Path("/home/zonghuan/tudelft/projects/datasets/conflab/annotations/f_formations"))
    # df.to_pickle("../data/export/data_dense_feats.pkl")
