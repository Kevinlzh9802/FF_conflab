%% Detect concurrent speakers over sliding window
window_bounds = [1, 20] * 60;  % [min, max] window size
step = 60;
w_ind = 1;
for n=1:30
    max_speaker{n} = repmat({0}, 20, 15);  % Creates a 3x4 cell array with 0 in each cell
end

for w = window_bounds(1):step:window_bounds(2)
    % Process all formations, not just those meeting window size requirement
    floors = cell(height(formations), 1);
    max_floors = cell(height(formations), 1);
    valid_indices = [];
    
    for i = 1:height(formations)
        if cellfun(@length, formations.timestamps_all(i)) >= w
            % Formation meets window size requirement - process normally
            actions = sp_merged{formations.Vid(i)};
            floors{i} = concurrent_speakers(formations(i,:), actions, w, step);
            max_floors{i} = max(floors{i}, [], "all");
            valid_indices = [valid_indices; i];
            
            card = formations.cardinality(i);
            max_speaker{card}{w_ind, max_floors{i}+1} = ...
                max_speaker{card}{w_ind, max_floors{i}+1} + 1;
        else
            % Formation doesn't meet window size requirement - count as max_speaker=0
            floors{i} = [];
            max_floors{i} = 0;
            
            card = formations.cardinality(i);
            max_speaker{card}{w_ind, 1} = ...
                max_speaker{card}{w_ind, 1} + 1;
        end
    end

    % Create table with id, cardinality, and floor data (only for valid formations)
    if ~isempty(valid_indices)
        filtered_fs = formations(valid_indices, :);
        filtered_floors = floors(valid_indices);
        filtered_max_floors = max_floors(valid_indices);
        out_table{w_ind, 1} = table(filtered_fs.id, filtered_fs.cardinality, ...
            filtered_floors, filtered_max_floors);
    else
        out_table{w_ind, 1} = table();
    end
    out_table{w_ind, 2} = w;
    % save(fullfile(outdir, sprintf('%d.mat', w)), 'out_table');
    w_ind = w_ind + 1;
    % x = out_table{w_ind,1};
end
run plotFloorsCustom.m;

%% Functions

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

