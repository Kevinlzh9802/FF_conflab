#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: detect_sub_f_floors.py
# Created Date: Tuesday, May 28th 2019, 2:13:31 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import argparse

from numpy.lib.stride_tricks import as_strided as stride
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# import constants.paths as paths


def _annotation_slice_for_formation(row, actions, annotation):
    """ Return the actions df slice corresponding to the F-formation """
    participants = list(map(int, row["participants"].split()))
    day = int(row.name[-1])
    idx = pd.IndexSlice
    start, end = int(row["sample_start"]), int(row["sample_end"])
    data = actions.loc[start:end, idx[day, participants, annotation]]
    return data


def _roll(df, window, step, **kwargs):
    """ Roll over the dataframe """
    val = df.values
    d0, d1 = val.shape
    s0, s1 = val.strides

    a = stride(val, (((d0 - window) // step) + 1, window, d1), (s0 * step, s0, s1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)


def _concurrent_speakers(formation, actions, window_size, step_size):
    """ Calculate concurrent speakers over a rolling window in an F-formation.

    Args:
        formation       : A row in the F-formations data frame
        actions         : The annotation data frame with speaker status.
        window_size     : Threshold for speaking duration
        step_size       : Stride for moving threshold window for speaking duration

    Returns:
        The data frame of number of concurrent speakers for each rolling window

    """
    # Speaking status data slice for F-formation
    data = _annotation_slice_for_formation(formation, actions, "Speaking")

    # Lambda for calculating number of concurrent speakers for the windowed df
    n_speakers = lambda df: (df == 1).all().sum()

    # Calculate concurrent speakers for rolling window
    concurrent_speakers = _roll(data, window_size, step_size).apply(n_speakers)

    return concurrent_speakers


def detect_concurrent_speakers(formations, actions, window_bounds, step_size,
                               outdir):
    """ Compute the number of concurrent speakers for a given window size.

    Args:
        formations      : The dataframe of F-formations
        actions         : The dataframe of social action annotations
        window_size     : The (min, max) sliding window size for the duration
                          of which a person must be speaking. Both included
        step_size       : Step size for window duration
        outdir          : The Pathlib directory to write processed data frames

    """
    for w in tqdm(range(window_bounds[0], window_bounds[1]+1, step_size)):
        filtered_fs = formations[
            formations.sample_end - formations.sample_start >= w
        ]
        floors = filtered_fs.apply(
            _concurrent_speakers, args=(actions, w, step_size), axis=1
        )
        floors = pd.concat([filtered_fs.id, filtered_fs.cardinality, floors], axis=1)
        floors.to_pickle(outdir / (str(w) + ".pkl"))


def check_if_lost(row, actions, annotation):
    """ Check if any of the participants were lost in the time segment.

    Args:
        row             : The row in the F-formations annotation frame
        actions         : The annotations dataframe (assumes columns of
                          [day, participant, annotation])
        annotation      : The list of action labels of interest

    Returns:
        True if any of the entries in the dataframe subset is -1

    """
    data = _annotation_slice_for_formation(row, actions, annotation)
    return (data == -1).any(axis=None)


def main():
    """ Main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("social_actions",
                        help="Pickle file with social action annotations ")
    parser.add_argument("f_formations",
                        help="Pickle file with F-formation annotations ")
    parser.add_argument("outdir", help="Directory to write dataframes to")
    parser.add_argument("window_bounds", nargs=2, type=int,
                        help="Window size bounds")
    parser.add_argument("step", type=int, help="Window size step")
    args = parser.parse_args()

    # Load social action annotations and F-formations
    actions = pd.read_pickle(args.social_actions)
    formations = pd.read_pickle(args.f_formations)

    # Add column for number of F-fomration members
    formations["cardinality"] = formations["participants"].str.split().str.len()

    # Add ids for the f-formations
    formations["id"] = range(1, 1+len(formations))

    # Filter F-formations where person might be missing.
    formations["missing"] = formations.apply(
        check_if_lost, args=(actions, "Speaking"), axis=1
    )
    formations = formations[~formations.missing]

    # Filter F-formations with cardinality less than 4
    formations = formations[formations.cardinality >= 4]

    # Create the output directory and detect conversation floors
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    detect_concurrent_speakers(
        formations, actions, args.window_bounds, args.step, outdir
    )


if __name__ == "__main__":
    main()