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
from pathlib import Path


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

def fit_GLM(formula, data):
    """ Fits a GLM to the data following formula """
    model = smf.glm(formula=formula, data=data,  family=sm.families.Poisson()).fit()
    # print(model.summary())
    return model


def main():
    """ Compare drop-offs """
    """ Main function """
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_floors", help="DataFrame of max conversation floors", 
                        default='./data/max_floors_data.csv')
    parser.add_argument("--outdir", help="Directory to write data and plots to", 
                        default='./data/glm_results')
    args = parser.parse_args()

    # Create the output directory and detect conversation floors
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(args.max_floors)

    # Filter Groups 140, 142, 153 because of incorrect annotations
    # data = data[~data.id.isin([89, 140, 142, 153])]
    
    # Filter data to use cardinality 4-7 only
    data = data[data['cardinality'].between(4, 7)]
    data['window_size'] = data['window_size'] / 60.0
    print(f"Filtered data to cardinalities 4-7. Remaining observations: {len(data)}")
    print(f"Cardinality distribution: {data['cardinality'].value_counts().sort_index()}")

    # Setup formula
    formula = "max_floors ~ window_size * cardinality"

    # Fit the model
    model = fit_GLM(formula, data)
    print(model.summary())
    model.save(outdir / "single_glm.pkl")
    
    # Fit pairwise models between each pair of cardinalities
    print("\nFitting pairwise GLM models between cardinality pairs...")

    for pair in [(4,5), (4,6), (4,7), (5,6), (5,7), (6,7)]:
        subset = data[data.cardinality.isin(pair)]
        model_pair = fit_GLM("max_floors ~ window_size * cardinality", subset)
        print(f"Pair {pair}")
        print(model_pair.pvalues)


if __name__ == "__main__":
    main()
    