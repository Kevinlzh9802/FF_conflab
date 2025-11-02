import os
import pandas as pd
import numpy as np
import json

from utils.pose import process_foot_data

ALL_CLUES = ["head", "shoulder", "hip", "foot"]

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
    df = df['row_id Cam Vid Seg concat_ts pixelCoords spaceCoords pixelFeat spaceFeat GT'.split()]
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


if __name__ == '__main__': 
    process_data("../data/export/")
