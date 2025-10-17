% Construct formations for a given base_clue
% Inputs:
%   base_clue - string indicating the clue type (e.g., "head", "foot", "hip", "shoulder", "GT")
%   data_results - table containing the detection results
%   speaking_status - structure containing speaking status data
%   outdir - output directory path

% Equivalent MATLAB script for the Python `main` function

% Collect data for all clues first
all_max_speaker_data = struct();
clues_to_process = ["GT"];

for base_clue = clues_to_process
    if base_clue == "GT"
        col_name = "GT";
    else
        col_name = base_clue + "Res";
    end
    unique_segs = unique(data_results.Vid);
    formations = table;
    for u_ind=1:length(unique_segs)
        u = unique_segs(u_ind);
        ana = data_results(data_results.Vid == u, :);
        unique_groups = recordUniqueGroups(ana, col_name);
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
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    %% Average speaker
    for i = 1:height(formations)
        ts = formations.timestamps_all{i};
        participants = formations.participants{i};
        % Ensure participants is always a row vector
        if iscolumn(participants)
            participants = participants';
        end
        seg = formations.Vid(i);

        % Find column indices where first row matches participant IDs
        all_participants = sp_merged{seg}(1, :);
        participant_cols = [];
        for p = participants
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
     % Run detectSubFloor for this base_clue and collect max_speaker data
     % max_speaker = detectSubFloor(formations, sp_merged, base_clue, outdir);
     
     % Store max_speaker data for this clue
     % all_max_speaker_data.(base_clue) = max_speaker;
     
     % Plot cumulative F-formation counts for GT case
     if strcmp(base_clue, 'GT')
         fprintf('Creating cumulative F-formation plot...\n');
         % plotSamplesPerWindowSize(formations);
     end
end

%% Functions
% Equivalent of check_if_lost
function lost = check_if_lost(row, actions)
data = annotation_slice_for_formation(row, actions);
lost = any(data(2:end, :) == -1, 'all');
end

% Equivalent of _annotation_slice_for_formation
function data = annotation_slice_for_formation(row, actions)
    participants = row.participants{1}; % converts space-separated string to numeric array
    % Ensure participants is always a row vector
    if iscolumn(participants)
        participants = participants';
    end
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

function plotSamplesPerWindowSize(formations)
    cardinalities = 4:7;
    window_sizes = 1:20;
    
    % Calculate cumulative counts for each cardinality
    cumulative_counts = zeros(length(window_sizes), length(cardinalities));
    
    for c_idx = 1:length(cardinalities)
        cardinality = cardinalities(c_idx);
        for w_idx = 1:length(window_sizes)
            window_size = window_sizes(w_idx);
            % Count formations with cardinality and window_size >= current window
            % Apply length() to each row's timestamps
            cardinality_mask = formations.cardinality == cardinality;
            window_lengths = cellfun(@length, formations.timestamps);
            count = sum(cardinality_mask & window_lengths >= window_size);
            cumulative_counts(w_idx, c_idx) = count;
        end
    end
    
    % Create step plot similar to the paper's figure
    figure('Position', [100, 100, 1200, 600]);
    colors = [0, 0, 1; 1, 0.5, 0; 0, 0.8, 0; 1, 0, 0]; % Blue, Orange, Green, Red
    
    hold on;
    for c_idx = 1:length(cardinalities)
        cardinality = cardinalities(c_idx);
        plot(window_sizes, cumulative_counts(:, c_idx), 'Color', colors(c_idx, :), ...
            'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4, ...
            'DisplayName', sprintf('Cardinality %d', cardinality));
    end
    
    xlabel('Speaking Duration (seconds)', 'FontSize', 12);
    ylabel('Number of F-formations', 'FontSize', 12);
    title('Number of F-formation Samples at Varying Turn Durations', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Set axis properties
    xlim([0, 21]);
    ylim([0, max(cumulative_counts(:)) + 1]);
    xticks(0:1:20);
    grid on;
    grid minor;
    
    % Add legend
    legend('Location', 'southwest', 'FontSize', 10);
    
    % Print summary statistics
    fprintf('Cumulative F-formation counts by cardinality:\n');
    fprintf('==========================================\n');
    for c_idx = 1:length(cardinalities)
        cardinality = cardinalities(c_idx);
        total_formations = cumulative_counts(1, c_idx); % Count at window_size = 1
        fprintf('Cardinality %d: %d total formations\n', cardinality, total_formations);
    end
end