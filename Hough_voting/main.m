clear variables; 
% clearvars -except frames;
close all;
warning off;

load('../data/head.mat', 'all_data');
% load('../data/filtered/frames.mat', 'frames');
addpath(genpath('../utils'));

used_data = filterTable(all_data, 'all', [2,3], 'all');
GTgroups = (used_data.GT)';

params.r = 100;               % o-space radius
params.sigma_pos = 10;
params.sigma_ang = 1;
params.N = 800;
params.grid_size = [1920, 1080];
params.tau_intr = 0.7;
params.peak_thresh = 1e-7;

all_groups = [];
for k=200:200
    feature = used_data.Features{k};
    all_groups{k} = detect_f_formations(feature, params);
    % all_groups{k}.members
    % all_groups{k}.center
end

function groups = detect_f_formations(feature, params)
% Main pipeline
person_ids = feature(:, 1);
positions = feature(:, 2:3);
orientations = feature(:, 4);
num_person = size(person_ids, 1);

[samples, weights] = generate_samples(positions, orientations, ...
    params.sigma_pos, params.sigma_ang, params.N);

labels = repelem(1:num_person, params.N)';
[AI, AL, sz] = vote_for_o_space(samples, weights, labels, num_person, ...
    params.r, params.grid_size);
Atilde = compute_accumulator(AI, AL);

groups = [];
center_count = 0;
max_val_1 = max(Atilde(:));
while true
    [max_val, idx] = max(Atilde(:));
    if max_val < params.peak_thresh || center_count > num_person
        break;
    end
    [row, col] = ind2sub(sz, idx);
    members = find(AL(row, col, :) > 0);
    center = [col, row];

    is_valid = validate_o_space(center, samples, labels, members, weights, params.r, params.tau_intr);
    if is_valid
        groups(end+1).members = person_ids(members);
        groups(end).center = center;
        center_count = center_count + 1;
    end
    imshow(Atilde / max_val_1);
    Atilde(row, col) = 0; % suppress this peak
end
end
