import json, os
import pandas as pd
import numpy as np
from utils.scripts import concat_segs
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

def filter_and_concat_table(data, used_segs=None): #TODO: move to table.py
    df = data
    if isinstance(df, pd.DataFrame) and used_segs:
        sel = df.iloc[0:0].copy()
        for key in used_segs:
            if isinstance(key, str) and len(key) == 3 and key.isdigit():
                cam, vid, seg = int(key[0]), int(key[1]), int(key[2])
                mask = (df['Cam'] == cam) & (df['Vid'] == vid) & (df['Seg'] == seg)
                if mask.any():
                    sel = pd.concat([sel, df.loc[mask]], ignore_index=True)
        if len(sel) > 0:
            df = sel
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

if __name__ == '__main__': 
    process_data("../data/export/")
