clear variables; 
% clearvars -except frames;
close all;
warning off;

load('../data/filtered/head.mat', 'all_data');
% load('../data/filtered/frames.mat', 'frames');
addpath(genpath('../utils'));

used_data = filterTable(all_data, 'all', [2,3], 'all');
GTgroups = (used_data.GT)';
features = (used_data.Features)';

params.r = 0.75;               % o-space radius
params.sigma_pos = 0.2;
params.sigma_ang = 0.05;
params.N = 800;
params.grid_size = 0.1;
params.tau_intr = 0.7;
params.peak_thresh = 5;

detect_f_formations();



function groups = detect_f_formations(positions, orientations, params)
% Main pipeline
[samples, weights] = generate_samples(positions, orientations, ...
    params.sigma_pos, params.sigma_ang, params.N);

labels = repelem(1:size(positions,1), params.N)';
[AI, AL, sz] = vote_for_o_space(samples, weights, labels, params.r, params.grid_size);
Atilde = compute_accumulator(AI, AL);

groups = [];
while true
    [max_val, idx] = max(Atilde(:));
    if max_val < params.peak_thresh
        break;
    end
    [row, col] = ind2sub(sz, idx);
    members = AL{row, col};
    center = [col, row];

    is_valid = validate_o_space(center, samples, labels, members, weights, params.r, params.tau_intr);
    if is_valid
        groups(end+1).members = members;
        groups(end).center = center;
    end

    Atilde(row, col) = 0; % suppress this peak
end
end
