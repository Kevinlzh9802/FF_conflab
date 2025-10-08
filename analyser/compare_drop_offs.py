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
    
    # # Get unique cardinalities
    # unique_cardinalities = sorted(data['cardinality'].unique())
    # print(f"Cardinalities found: {unique_cardinalities}")
    
    # # Create all possible pairs
    # pairwise_results = []
    # pair_count = 0
    
    # for i in range(len(unique_cardinalities)):
    #     for j in range(i + 1, len(unique_cardinalities)):
    #         card1, card2 = unique_cardinalities[i], unique_cardinalities[j]
            
    #         # Filter data for this pair
    #         pair_data = data[data['cardinality'].isin([card1, card2])].copy()
            
    #         if len(pair_data) < 4:  # Need at least 4 observations for meaningful GLM
    #             print(f"Skipping pair ({card1}, {card2}): insufficient data ({len(pair_data)} observations)")
    #             continue
            
    #         print(f"Fitting GLM for pair ({card1}, {card2}) with {len(pair_data)} observations...")
            
    #         try:
    #             # Fit GLM for this pair
    #             formula = "max_floors ~ window_size * cardinality"
    #             model = fit_GLM(formula, pair_data)
                
    #             # Extract p-values for interaction term (window_size:cardinality)
    #             # This tests if the effect of window_size differs between cardinalities
    #             pvalue_interaction = model.pvalues.get('window_size:cardinality', model.pvalues.get('window_size*cardinality', None))
                
    #             if pvalue_interaction is None:
    #                 # If interaction term not found, use the main effect p-value
    #                 pvalue_interaction = model.pvalues.get('window_size', 1.0)
                
    #             pairwise_results.append({
    #                 'cardinality_1': card1,
    #                 'cardinality_2': card2,
    #                 'n_obs': len(pair_data),
    #                 'pvalue_interaction': pvalue_interaction,
    #                 'pvalue_window_size': model.pvalues.get('window_size', 1.0),
    #                 'pvalue_cardinality': model.pvalues.get('cardinality', 1.0)
    #             })
                
    #             pair_count += 1
                
    #         except Exception as e:
    #             print(f"Error fitting GLM for pair ({card1}, {card2}): {e}")
    #             continue
    
    # # Convert to DataFrame
    # pairwise_df = pd.DataFrame(pairwise_results)
    
    # if not pairwise_df.empty:
    #     # Apply Bonferroni correction
    #     n_comparisons = len(pairwise_df)
    #     pairwise_df['pvalue_interaction_corrected'] = pairwise_df['pvalue_interaction'] * n_comparisons
    #     pairwise_df['pvalue_interaction_corrected'] = pairwise_df['pvalue_interaction_corrected'].clip(upper=1.0)
        
    #     # Save results
    #     pairwise_df.to_csv(outdir / "pairwise_glm_results.csv", index=False)
    #     pairwise_df.to_pickle(outdir / "pairwise_glm_results.pkl")
        
    #     print(f"\nPairwise GLM Results (n={n_comparisons} comparisons):")
    #     print("=" * 80)
    #     print(pairwise_df[['cardinality_1', 'cardinality_2', 'n_obs', 
    #                       'pvalue_interaction', 'pvalue_interaction_corrected']].round(4))
        
    #     # Summary of significant differences
    #     significant_pairs = pairwise_df[pairwise_df['pvalue_interaction_corrected'] < 0.05]
    #     print(f"\nSignificant differences (p < 0.05 after Bonferroni correction): {len(significant_pairs)}")
    #     if len(significant_pairs) > 0:
    #         print(significant_pairs[['cardinality_1', 'cardinality_2', 'pvalue_interaction_corrected']].round(4))
    # else:
    #     print("No pairwise comparisons could be performed.")


if __name__ == "__main__":
    main()