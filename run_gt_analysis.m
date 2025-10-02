% Run GT analysis and generate data for Python GLM analysis
% This script runs the formation construction and subfloor detection for GT case only

clear; clc;

% Set up paths
outdir = './glm_results';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% Load data (assuming data_results and speaking_status are already loaded)
% You may need to modify these paths based on your data location
if ~exist('data_results', 'var')
    fprintf('Loading data_results...\n');
    % Add your data loading code here
    % load('path/to/your/data_results.mat');
end

if ~exist('speaking_status', 'var')
    fprintf('Loading speaking_status...\n');
    % Add your speaking status loading code here
    % load('path/to/your/speaking_status.mat');
end

% Run the analysis for GT case only
fprintf('Running GT analysis...\n');
constructFormations(data_results, speaking_status, outdir);

fprintf('GT analysis completed!\n');
fprintf('Data files saved in: %s\n', outdir);
fprintf('Files generated:\n');
fprintf('  - max_floors_data.mat\n');
fprintf('  - max_floors_data.csv\n');
fprintf('\nNext steps:\n');
fprintf('1. Run: python analyser/run_glm_analysis.py --outdir %s\n', outdir);
fprintf('2. Or run the conversion and GLM analysis separately:\n');
fprintf('   python analyser/load_matlab_data.py --csv_file %s/max_floors_data.csv --outdir %s\n', outdir, outdir);
fprintf('   python analyser/compare_drop_offs.py %s/max_floors_data.pkl %s\n', outdir, outdir);
