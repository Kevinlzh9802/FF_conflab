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


base_clue = "hip";
unique_segs = unique(data_results.Vid);
formations = table;
for u_ind=1:length(unique_segs)
    u = unique_segs(u_ind);
    ana = data_results(data_results.Vid == u, :);
    unique_groups = recordUniqueGroups(ana, base_clue + "Res");
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
% f_name = "vid" + params.vids + "_seg" + params.segs;
% actions = speaking_status.speaking.(f_name);

% Create new formations table with all valid subsequences
new_formations = table;
min_subseq_length = 3; % Minimum length for a subsequence to be considered

for i = 1:height(formations)
    % Find all valid subsequences for this formation
    all_subseqs = all_connected_subseqs(formations.timestamps{i}, 61, min_subseq_length);
    
    for j = 1:length(all_subseqs)
        % Create a new entry for each valid subsequence
        new_row = formations(i, :);
        new_row.timestamps = {all_subseqs{j}};
        new_row.longest_ts = {all_subseqs{j}};
        new_row.timestamps_all = {generate_continuous_sequence(all_subseqs{j}, 61)};
        new_row.missing = check_if_lost(new_row, sp_merged{formations.Vid(i)});
        
        % Only add if not missing data
        if ~new_row.missing
            new_formations = [new_formations; new_row];
        end
    end
end

% Replace original formations with new ones
formations = new_formations;

% Update ID column for the new formations
formations.id = (1:height(formations))';

% Filter out rows with missing data and with fewer than 4 participants
formations = formations(~formations.missing, :);
formations = formations(formations.cardinality >= 4, :);
formations = formations(cellfun(@length, formations.timestamps_all) > 0, :);
formations.Cam = cell2mat(formations.Cam);
% Create output directory if it doesn’t exist
% if ~exist(outdir, 'dir')
%     mkdir(outdir);
% end

%% Average speaker
for i = 1:height(formations)
    ts = formations.timestamps_all{i};
    participants = formations.participants{i};
    seg = formations.Vid(i);
    
    % Find column indices where first row matches participant IDs
    all_participants = sp_merged{seg}(1, :);
    participant_cols = [];
    for p = participants'
        col_idx = find(all_participants == p);
        if ~isempty(col_idx)
            participant_cols = [participant_cols, col_idx];
        end
    end
    sp_participants = sp_merged{seg}(ts+1, participant_cols);
    formations.avg_speaker(i) = sum(sp_participants,"all") / ...
        length(sp_participants);
end

% Print average speaker values for each cardinality
fprintf('Average speaker values by cardinality:\n');
for card = 4:7
    card_formations = formations(formations.cardinality == card, :);
    if ~isempty(card_formations)
        avg_value = mean(card_formations.avg_speaker);
        fprintf('Cardinality %d: %.4f (n=%d)\n', card, avg_value, height(card_formations));
    else
        fprintf('Cardinality %d: No formations found\n', card);
    end
end
fprintf('\n');
run detectSubFloor.m;

%% Functions
% Equivalent of check_if_lost
function lost = check_if_lost(row, actions)
    data = annotation_slice_for_formation(row, actions);
    lost = any(data(2:end, :) == -1, 'all');
end

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

function all_subseqs = all_connected_subseqs(seq, n, min_length)
% Finds all contiguous subsequences in `seq`
% such that the difference between adjacent elements ≤ n
% and the subsequence length >= min_length

    all_subseqs = {};
    start_idx = 1;

    for i = 2:length(seq)
        if seq(i) - seq(i-1) <= n
            % Continue the current subsequence
            continue;
        else
            % End of a valid segment
            len = i - start_idx;
            if len >= min_length
                subseq = seq(start_idx : i-1);
                all_subseqs{end+1} = subseq;
            end
            start_idx = i;  % reset
        end
    end

    % Check the final subsequence
    len = length(seq) - start_idx + 1;
    if len >= min_length
        subseq = seq(start_idx : end);
        all_subseqs{end+1} = subseq;
    end
end