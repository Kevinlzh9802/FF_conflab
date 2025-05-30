#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: compare_drop_offs.py
# Created Date: Wednesday, June 12th 2019, 1:25:15 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import argparse

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_GLM_to_group(group):
    """ Fits a Poisson Family GLM and returns the parameters and z-scores """
    g = group.sort_values(by="window_size")
    cardinality = g.cardinality.values[0]
    x = sm.add_constant((g.window_size / 20).values)
    y = g.max_floors.values
    poisson_model = sm.GLM(y, x, family=sm.families.NegativeBinomial())
    results = poisson_model.fit()
    zs = results.params / results.bse
    return pd.Series({
        "a":results.params[1], "b":results.params[0], "az":zs[1], "bz":zs[0],
        "cardinality":cardinality
    })

def fit_GLM(data, formula):
    """ Fits a GLM to the data following formula """
    model = smf.glm(formula=formula, data=data,  family=sm.families.Poisson()).fit()
    print(model.summary())
    return model


def main():
    """ Compare drop-offs """
    """ Main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("max_floors",
                        help="DataFrame of max conversation floors")
    parser.add_argument("outdir",
                        help="Directory to write data and plots to")
    args = parser.parse_args()

    # Create the output directory and detect conversation floors
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_pickle(args.max_floors)

    # Filter Groups 140, 142, 153 because of incorrect annotations
    data = data[~data.id.isin([89, 140, 142, 153])]

    # Setup formula
    formula = "max_floors ~ window_size * cardinality"

    # Fit the model
    model = fit_GLM(formula, data)
    model.save(outdir / "single_glm.pkl")

    # Fit pairwise models

if __name__ == "__main__":
    main()