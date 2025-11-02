#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Load MATLAB data and convert to format expected by compare_drop_offs.py
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path



def load_csv_data(csv_file):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def convert_to_python_format(df):
    """Convert MATLAB data to format expected by compare_drop_offs.py"""
    if df is None or df.empty:
        print("No data to convert")
        return None
    
    # Ensure we have the required columns
    required_cols = ['cardinality', 'window_size', 'max_floors']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None
    
    # Convert window_size to the format expected by the Python script
    # The Python script expects window_size to be divided by 20
    df_converted = df.copy()
    df_converted['window_size'] = df_converted['window_size'] / 20.0
    
    # Ensure max_floors is integer
    df_converted['max_floors'] = df_converted['max_floors'].astype(int)
    
    # Remove any rows with invalid data
    df_converted = df_converted.dropna()
    df_converted = df_converted[df_converted['cardinality'] >= 4]  # Filter cardinality >= 4
    
    return df_converted


def main():
    """Main function to convert MATLAB data for Python analysis"""
    parser = argparse.ArgumentParser(description='Convert MATLAB data for Python GLM analysis')
    parser.add_argument('--csv_file', help='Path to CSV file', default='./data/max_floors_data.csv')
    parser.add_argument('--output', help='Output pickle file path', default='max_floors_data.pkl')
    parser.add_argument('--outdir', help='Output directory', default='./data/')
    
    args = parser.parse_args()
    
    # Load data
    df = None
    if args.csv_file and Path(args.csv_file).exists():
        print(f"Loading data from CSV: {args.csv_file}")
        df = load_csv_data(args.csv_file)
    else:
        print("No valid input file provided")
        return
    
    if df is None:
        print("Failed to load data")
        return
    
    # Convert to Python format
    df_converted = convert_to_python_format(df)
    if df_converted is None:
        print("Failed to convert data")
        return
    
    # Save converted data
    output_path = Path(args.outdir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_converted.to_pickle(output_path)
    print(f"Saved converted data to: {output_path}")
    print(f"Data shape: {df_converted.shape}")
    print(f"Columns: {list(df_converted.columns)}")
    print(f"Cardinalities: {sorted(df_converted['cardinality'].unique())}")
    print(f"Window sizes: {sorted(df_converted['window_size'].unique())}")
    print(f"Max floors range: {df_converted['max_floors'].min()} to {df_converted['max_floors'].max()}")


if __name__ == "__main__":
    main()
