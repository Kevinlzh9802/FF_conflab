#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test script to verify data format for GLM analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path


def test_data_format(csv_file):
    """Test if the data format is correct for GLM analysis"""
    print(f"Testing data format in: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['id', 'cardinality', 'window_size', 'max_floors']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        else:
            print("✅ All required columns present")
        
        # Check data types and ranges
        print("\nData summary:")
        print(f"  ID range: {df['id'].min()} to {df['id'].max()}")
        print(f"  Cardinalities: {sorted(df['cardinality'].unique())}")
        print(f"  Window sizes: {sorted(df['window_size'].unique())}")
        print(f"  Max floors range: {df['max_floors'].min()} to {df['max_floors'].max()}")
        
        # Check for missing values
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            print(f"❌ Missing values found: {missing_count[missing_count > 0].to_dict()}")
            return False
        else:
            print("✅ No missing values")
        
        # Check cardinality >= 4 (as expected by the original script)
        low_card = df[df['cardinality'] < 4]
        if len(low_card) > 0:
            print(f"⚠️  Found {len(low_card)} formations with cardinality < 4")
        else:
            print("✅ All formations have cardinality >= 4")
        
        # Check window size format (should be in samples, not divided by 20)
        window_sizes = df['window_size'].unique()
        if all(ws >= 60 for ws in window_sizes):  # Assuming window sizes are in samples
            print("✅ Window sizes appear to be in samples (not divided by 20)")
        else:
            print("⚠️  Window sizes might need conversion")
        
        print("\n✅ Data format looks good for GLM analysis!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def main():
    """Test the data format"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data format for GLM analysis')
    parser.add_argument('csv_file', help='Path to CSV file to test')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"❌ File not found: {args.csv_file}")
        return
    
    success = test_data_format(args.csv_file)
    
    if success:
        print("\n🎉 Data is ready for GLM analysis!")
        print("You can now run:")
        print("  python load_matlab_data.py --csv_file <csv_file> --outdir <outdir>")
        print("  python compare_drop_offs.py <pkl_file> <outdir>")
    else:
        print("\n❌ Data format issues found. Please check the MATLAB output.")


if __name__ == "__main__":
    main()
