%% Detect how many groups simultaneous speakers belong to
% This script analyzes the entire speaking status for all people
% and determines how many groups simultaneous speakers belong to

% Parameters
window_bounds = [1, 20] * 60;  % [min, max] window size
step = 60;

% Initialize results storage
group_num_results = cell(length(window_bounds(1):step:window_bounds(2)), 1);
w_ind = 1;

% Process each video separately
% Analyze each window and create table
window_table = table();
for vid = 2:3  % Assuming videos 2 and 3 as in the original code
    fprintf('Processing video %d...\n', vid);

    % Process each window size
    for w = window_bounds(1):step:window_bounds(2)
        fprintf('  Window size: %d frames\n', w);

        % Roll over the entire speaking status for all people
        rolled_data = roll_df_all_people(sp_merged{vid}, w, step);

        for i = 1:length(rolled_data)
            window_data = rolled_data{i};

            % Store speaking status for each person at each time position in this window
            speaking_status_window = store_speaking_status_per_window(window_data, i, step);

            % Create row for table
            row = table();
            row.id = i;
            row.Vid = vid;
            row.time = {[speaking_status_window.start_time, speaking_status_window.end_time]};
            row.length = speaking_status_window.window_length;
            row.speaking_all_time = {speaking_status_window.speaking_all_time};

            % Append to table
            window_table = [window_table; row];
        end

        % Store results for this window size
        group_num_results{w_ind} = struct('video', vid, ...
                                        'window_size', w, ...
                                        'window_table', window_table);
        w_ind = w_ind + 1;
    end
end

% display_group_num_summary(group_num_results);
window_table = countSpeakerGroups(window_table, data_results, ...
    {'headRes', 'shoulderRes', 'hipRes', 'footRes'}, 'no_aggregation');
[filtered_table, pairwise_diffs] = processWindowTable(window_table, 'no_aggregation');

%% Functions

% Roll over entire speaking status for all people
function rolled = roll_df_all_people(actions, window, step)
    % actions: matrix where first row contains participant IDs, 
    %          subsequent rows contain speaking status (0/1) for each time frame
    
    speaking_data = actions(2:end, :);  % Remove header row with IDs
    [d0, d1] = size(speaking_data);
    n_windows = floor((d0 - window) / step) + 1;
    rolled = cell(n_windows, 1);

    for i = 1:n_windows
        idx = (i-1)*step + (1:window);
        if idx(end) <= d0  % Ensure we don't go beyond data bounds
            window_data = [actions(1, :); speaking_data(idx, :)];  % Include ID row
            rolled{i} = window_data;
        end
    end
end

% Store speaking status for each person at each time position in a window
function speaking_status_window = store_speaking_status_per_window(window_data, window_idx, step)
    % window_data: matrix with first row as participant IDs, 
    %              subsequent rows as speaking status (0/1) for each time frame
    % window_idx: index of the current window
    % step: step size for window calculation
    
    participant_ids = window_data(1, :);
    speaking_status = window_data(2:end, :);
    
    % Create a structure to store simplified speaking status
    speaking_status_window = struct();
    
    % 1. Window length
    speaking_status_window.window_length = size(speaking_status, 1);
    
    % 2. Start and end time (in frame indices)
    start_time = (window_idx - 1) * step + 1;
    end_time = start_time + speaking_status_window.window_length - 1;
    speaking_status_window.start_time = start_time;
    speaking_status_window.end_time = end_time;
    
    % 3. Person IDs of those who spoke for ALL time during the window
    % A participant is considered "speaking for ALL time" if they spoke in every frame
    continuous_speakers = [];
    
    for p_idx = 1:length(participant_ids)
        participant_speaking = speaking_status(:, p_idx);
        % Check if this participant spoke in ALL frames (all values are 1)
        if all(participant_speaking == 1)
            continuous_speakers = [continuous_speakers, participant_ids(p_idx)];
        end
    end
    
    speaking_status_window.speaking_all_time = continuous_speakers;
    speaking_status_window.participant_ids = participant_ids;
end
