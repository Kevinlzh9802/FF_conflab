#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: plot_floors.py
# Created Date: Tuesday, June 4th 2019, 11:37:31 am
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import argparse

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_pubfig():
    sns.set_context("paper", font_scale=2, rc={
        "font.size":22, "axes.titlesize":22, "axes.labelsize":20,
        "lines.linewidth":2
    })
    sns.set_style({"font.family": "Times New Roman"})


def _max_windowed_floors(path):
    """ Load the conversation floors for the dataframe at `path`.

    Args:
        path    : The Pathlib path to the pickled dataframe

    Returns:
        The loaded dataframe with a column for `window_size` and max number of
        conversation floors in column `max_floors`.

    """
    df = pd.read_pickle(path)
    df["max_floors"] = df.iloc[:, ~df.columns.isin(["cardinality", "id"])].max(axis=1)
    df["window_size"] = int(path.stem)
    return df[["id","cardinality", "window_size", "max_floors"]]


def plot_floors(df):
    """ Plots the facet grid """
    df["window_size"] = df["window_size"] / 20
    g = sns.relplot(
        x="window_size", y="max_floors", col="cardinality",
        data=df, kind="line", ci="sd", err_style="bars", height=12, aspect=1.2
    )
    g.set(xticks=np.arange(1,21,1))
    g.set(xlim=(0,None))
    g.fig.suptitle("Variation in Number of Conversation Floors with Speaking Duration Threshold")
    g.set_axis_labels(
        x_var="Speaking Duration in seconds",
        y_var="Mean (Max. No. of Distinct Floors) per F-formation")
    plt.subplots_adjust(top=0.85)
    return g


def plot_cardinality_counts(df):
    """ Plot number of samples per window_size by cardinality """
    df["window_size"] = df["window_size"] / 20
    ax = (df.groupby(["cardinality"]).window_size.value_counts()
        .unstack("cardinality")
        .plot(sharex=True)
    )
    ax.set(yticks=np.arange(0,22,1), xticks=np.arange(1,21,1))
    ax.set(
        xlabel="Speaking Duration in seconds",
        ylabel="Number of F-formations",
        title="Number of F-formation Samples at Varying Turn Durations"
    )
    return ax.get_figure()


def main():
    """ Main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("indir",
                        help="Directory with dataframes of conversation floors")
    parser.add_argument("outdir",
                        help="Directory to write plots to")
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    # Iterate over pickled dataframes and construct master frame
    dfs = []
    for path in indir.glob("*.pkl"):
        df = _max_windowed_floors(path)
        dfs.append(df)

    set_pubfig()

    data = pd.concat(dfs, ignore_index=True)
    # data.to_pickle(outdir / "summary_df.pkl")
    fig = plot_cardinality_counts(data)
    fig.savefig(outdir / "counts.png", bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    main()