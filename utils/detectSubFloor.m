% Equivalent MATLAB script for the Python `main` function

% Define the input arguments as global variables or manually set them here
% Replace the following with your actual paths and values
% social_actions_file = 'social_actions.pkl';
% f_formations_file = 'f_formations.pkl';
% 
% % Load the social actions and f-formations data (assuming .pkl files converted to .mat)
% load(social_actions_file, 'actions');  % variable 'actions' loaded
% load(f_formations_file, 'formations');  % variable 'formations' loaded

load('../data/speaking_status.mat', 'speaking_status');

outdir = 'output_directory';
window_bounds = [1, 20] * 60;  % [min, max] window size
step = 60;

unique_segs = unique(data_results.Vid);
formations = table;
for u_ind=1:length(unique_segs)
    u = unique_segs(u_ind);
    ana = data_results(data_results.Vid == u, :);
    unique_groups = recordUniqueGroups(ana, "GT");
    unique_groups.Vid = zeros(height(unique_groups), 1) + u;
    formations = [formations; unique_groups];
end

% merge speaking status first
sp_merged{2} = mergeSpeakingStatus(speaking_status.speaking.vid2_seg8, ...
    speaking_status.speaking.vid2_seg9);

sp_merged{3} = mergeSpeakingStatus(speaking_status.speaking.vid3_seg1, ...
    speaking_status.speaking.vid3_seg2);
for k=3:6
    seg_name = "vid3_seg" + k;
    sp_merged{3} = mergeSpeakingStatus(sp_merged{3}, ...
        speaking_status.speaking.(seg_name));
end

% Add cardinality column to formations (count number of participants per row)
formations.cardinality = cellfun(@(x) numel(x), formations.participants);

% Add ID column (1-based indexing)
formations.id = (1:height(formations))';

% Check for missing participants (any -1 entries in "Speaking")
formations.missing = false(height(formations),1);
f_name = "vid" + params.vids + "_seg" + params.segs;
% actions = speaking_status.speaking.(f_name);

for i = 1:height(formations)
    formations.longest_ts{i} = longest_connected_subseq(formations.timestamps{i}, 61);
    formations.timestamps_all{i} = generate_continuous_sequence(formations.longest_ts{i}, 61);
    formations.missing(i) = check_if_lost(formations(i,:), sp_merged{formations.Vid(i)});
end

% Filter out rows with missing data and with fewer than 4 participants
formations = formations(~formations.missing, :);
formations = formations(formations.cardinality >= 4, :);

% Create output directory if it doesn’t exist
% if ~exist(outdir, 'dir')
%     mkdir(outdir);
% end

% Detect concurrent speakers over sliding window
w_ind = 1;
for n=1:30
    max_speaker{n} = repmat({0}, 20, 15);  % Creates a 3x4 cell array with 0 in each cell
end
for w = window_bounds(1):step:window_bounds(2)
    filtered_fs = formations(cellfun(@length,formations.timestamps_all) >= w, :);
    
    floors = cell(height(filtered_fs), 1);
    max_floors = cell(height(filtered_fs), 1);
    for i = 1:height(filtered_fs)
        actions = sp_merged{filtered_fs.Vid(i)};
        floors{i} = concurrent_speakers(filtered_fs(i,:), actions, w, step);
        max_floors{i} = max(floors{i}, [], "all");

        card = filtered_fs.cardinality(i);
        max_speaker{card}{w_ind, max_floors{i}+1} = ...
            max_speaker{card}{w_ind, max_floors{i}+1} + 1;
    end

    % Create table with id, cardinality, and floor data
    out_table{w_ind, 1} = table(filtered_fs.id, filtered_fs.cardinality, ...
        floors, max_floors);
    out_table{w_ind, 2} = w;
    % save(fullfile(outdir, sprintf('%d.mat', w)), 'out_table');
    w_ind = w_ind + 1;
    % x = out_table{w_ind,1};
    % for n=1:height(x)
    %     card = 
    % end
end

run plotFloorsCustom.m;


% Equivalent of _annotation_slice_for_formation
function data = annotation_slice_for_formation(row, actions)
    participants = row.participants{1}; % converts space-separated string to numeric array
    time_inds = row.timestamps_all{1};
    all_participants = actions(1, :);
    
    cols = ismember(all_participants, participants);
    % start_idx = row.sample_start;
    % end_idx = row.sample_end;

    % Extract relevant data slice
    data = actions([1, time_inds+1], cols);
end

% Equivalent of _roll
function rolled = roll_df(df, window, step)
    [d0, d1] = size(df);
    n_windows = floor((d0 - window) / step) + 1;
    rolled = cell(n_windows, 1);

    for i = 1:n_windows
        idx = (i-1)*step + (1:window);
        rolled{i} = df(idx, :);
    end
end

% Equivalent of _concurrent_speakers
function concurrent = concurrent_speakers(row, actions, window_size, step_size)
    data = annotation_slice_for_formation(row, actions);
    rolled = roll_df(data(2:end, :), window_size, step_size);

    concurrent = zeros(length(rolled), 1);
    for i = 1:length(rolled)
        window_data = rolled{i};
        concurrent(i) = sum(all(window_data == 1, 1)); % All speakers in row are speaking
    end
end

% Equivalent of check_if_lost
function lost = check_if_lost(row, actions)
    data = annotation_slice_for_formation(row, actions);
    lost = any(data(2:end, :) == -1, 'all');
end

function output = generate_continuous_sequence(sequence, k)
% Generates a continuous sequence from an increasing vector
% If the difference between adjacent elements >= k, it breaks the sequence

    output = [];
    n = length(sequence);

    for i = 1:n-1
        if sequence(i+1) - sequence(i) < k
            output = [output, sequence(i):sequence(i+1)];  % append the range
        end
    end

    output = unique(output);  % remove duplicates if ranges overlap
end

function subseq = longest_connected_subseq(seq, n)
% Finds the longest contiguous subsequence in `seq`
% such that the difference between adjacent elements ≤ n

    max_len = 0;
    start_idx = 1;
    best_start = 1;

    for i = 2:length(seq)
        if seq(i) - seq(i-1) <= n
            % Continue the current subsequence
            continue;
        else
            % End of a valid segment
            len = i - start_idx;
            if len > max_len
                max_len = len;
                best_start = start_idx;
            end
            start_idx = i;  % reset
        end
    end

    % Check the final subsequence
    len = length(seq) - start_idx + 1;
    if len > max_len
        max_len = len;
        best_start = start_idx;
    end

    % Extract the best subsequence
    subseq = seq(best_start : best_start + max_len - 1);
end