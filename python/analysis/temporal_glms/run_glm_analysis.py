#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Run complete GLM analysis pipeline:
1. Convert MATLAB data to Python format
2. Run GLM analysis using compare_drop_offs.py
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_matlab_analysis(matlab_dir, outdir):
    """Run MATLAB analysis to generate data"""
    print("Running MATLAB analysis...")
    
    # This assumes the MATLAB script is already set up to run
    # and will generate the data files in the output directory
    print(f"MATLAB analysis should generate data in: {outdir}")
    print("Please ensure constructFormations.m has been run with base_clue='GT'")


def convert_data(csv_file, outdir):
    """Convert MATLAB data to Python format"""
    print("Converting MATLAB data to Python format...")
    
    convert_cmd = [
        sys.executable, "load_matlab_data.py",
        "--csv_file", csv_file,
        "--output", "max_floors_data.pkl",
        "--outdir", outdir
    ]
    
    try:
        result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        print("Data conversion successful:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Data conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def run_glm_analysis(pkl_file, outdir):
    """Run GLM analysis using compare_drop_offs.py"""
    print("Running GLM analysis...")
    
    glm_cmd = [
        sys.executable, "compare_drop_offs.py",
        pkl_file,
        outdir
    ]
    
    try:
        result = subprocess.run(glm_cmd, check=True, capture_output=True, text=True)
        print("GLM analysis successful:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"GLM analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main function to run complete pipeline"""
    parser = argparse.ArgumentParser(description='Run complete GLM analysis pipeline')
    parser.add_argument('--matlab_dir', help='Directory containing MATLAB scripts', default='.')
    parser.add_argument('--outdir', help='Output directory', default='./glm_results')
    parser.add_argument('--csv_file', help='Path to CSV file from MATLAB', 
                       default='./max_floors_data.csv')
    
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GLM Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Run MATLAB analysis (user needs to do this manually)
    print("\nStep 1: MATLAB Analysis")
    print("-" * 30)
    run_matlab_analysis(args.matlab_dir, outdir)
    
    # Step 2: Convert data
    print("\nStep 2: Data Conversion")
    print("-" * 30)
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("Please run the MATLAB analysis first to generate the data.")
        return
    
    if not convert_data(str(csv_path), str(outdir)):
        print("Data conversion failed. Stopping pipeline.")
        return
    
    # Step 3: Run GLM analysis
    print("\nStep 3: GLM Analysis")
    print("-" * 30)
    pkl_file = outdir / "max_floors_data.pkl"
    if not pkl_file.exists():
        print(f"Error: Pickle file not found: {pkl_file}")
        return
    
    if not run_glm_analysis(str(pkl_file), str(outdir)):
        print("GLM analysis failed.")
        return
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Results saved in: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
