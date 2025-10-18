from typing import Any

import pandas as pd

from utils.python.eval import compute_hic_matrix


def getHIC(used_data: pd.DataFrame) -> pd.DataFrame:
    """Populate a 'HIC' column by computing a head interaction consistency matrix.

    Mirrors GCFF/getHIC.m behavior:
    used_data.HIC = cell(height(used_data), 1);
    used_data.HIC{k} = computeHICMatrix(used_data.GT{k}, used_data.headRes{k});

    Unknown dependency: computeHICMatrix — TODO: provide implementation in utilities.
    """
    df = used_data.copy()
    if 'HIC' not in df.columns:
        df['HIC'] = [None] * len(df)
    for idx, row in df.iterrows():
        df.at[idx, 'HIC'] = compute_hic_matrix(row.get('GT'), row.get('headRes'))
    return df
