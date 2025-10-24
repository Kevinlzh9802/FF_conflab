import json, os
import pandas as pd
import numpy as np

ALL_CLUES = ["head", "shoulder", "hip", "foot"]

def load_all_data() -> pd.DataFrame:
    all_data = None
    for clue in ALL_CLUES:
        data = load_data(clue)
        if all_data is None:
            all_data = data
            all_data.rename(columns={"Features": clue+"Feat"}, inplace=True)
        else:
            all_data[clue+"Feat"] = data["Features"]
    return all_data

def load_data(clue: str) -> pd.DataFrame:
    base = "../data/export/" + clue

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