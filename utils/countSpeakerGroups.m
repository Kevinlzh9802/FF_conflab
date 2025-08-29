function window_table = countSpeakerGroups(window_table, group_detection_table, ...
    groups_column, aggregation_method)
% COUNTSpeakerGroups - Count how many groups continuous speakers belong to and add to window_table
%
% Inputs:
%   window_table - Table with columns: id, Vid, time, length, speaking_all_time
%   group_detection_table - Table with columns: concat_ts, and various group detection columns
%   groups_column - String specifying which column contains the group detection results
%                  (e.g., 'headRes', 'shoulderRes', 'hipRes', 'footRes')
%   aggregation_method - String specifying how to aggregate group detections:
%                       'closest_to_start' (default), 'majority', 'union', etc.
%
% Output:
%   window_table - Updated table with total_groups column added

if nargin < 4
    aggregation_method = 'closest_to_start';
end

% Add new columns to window_table
window_table.detection = cell(height(window_table), 1);
window_table.filtered_speakers = cell(height(window_table), 1);
window_table.num_filtered_speakers = zeros(height(window_table), 1);
window_table.total_groups = zeros(height(window_table), 1);

% Process each window
for i = 1:height(window_table)
    window_row = window_table(i, :);
    window_start = window_row.time{1}(1);
    window_end = window_row.time{1}(2);
    speaking_vector = window_row.speaking_all_time{1};
    vid = window_row.Vid;
    
    % Get continuous speakers (person IDs who spoke for all time)
    continuous_speakers = speaking_vector;
    
    if isempty(continuous_speakers)
        window_table.detection{i} = {};
        window_table.filtered_speakers{i} = [];
        window_table.total_groups(i) = 0; % No continuous speakers, so 0 groups
        window_table.num_filtered_speakers(i) = 0; % No filtered speakers
        continue;
    end
    
    % Get group detection results for this window period
    window_groups = getGroupsForWindow(group_detection_table, groups_column, window_start, window_end, vid, aggregation_method);
    
    % Step 1: Filter speakers to only those who appear in any of the detected groups
    filtered_speakers = filterSpeakersByGroups(continuous_speakers, window_groups);
    
    % Step 2: Count total groups that filtered speakers belong to
    total_groups = countTotalGroups(filtered_speakers, window_groups);
    
    % Store results in the window_table
    window_table.detection{i} = window_groups;
    window_table.filtered_speakers{i} = filtered_speakers;
    window_table.num_filtered_speakers(i) = length(filtered_speakers);
    window_table.total_groups(i) = total_groups;
end

% Reorder columns to match requested order
window_table = window_table(:, {'id', 'Vid', 'time', 'length', 'speaking_all_time', ...
    'detection', 'filtered_speakers', 'num_filtered_speakers', 'total_groups'});

end

function window_groups = getGroupsForWindow(group_detection_table, groups_column, window_start, window_end, vid, method)
% Get aggregated group information for a specific window period
%
% Inputs:
%   group_detection_table - Table with concat_ts, Vid, and various group detection columns
%   groups_column - String specifying which column contains the group detection results
%   window_start, window_end - Window boundaries
%   vid - Video ID to match
%   method - Aggregation method
%
% Output:
%   window_groups - Cell array of group memberships for the window

% Find detections within the window period and matching video ID
in_window = group_detection_table.concat_ts >= window_start & ...
           group_detection_table.concat_ts <= window_end & ...
           group_detection_table.Vid == vid;

if ~any(in_window)
    window_groups = {}; % No detections in window
    return;
end

window_detections = group_detection_table(in_window, :);

switch method
    case 'closest_to_start'
        % Use the detection closest to window start time
        [~, closest_idx] = min(abs(window_detections.concat_ts - window_start));
        window_groups = window_detections.(groups_column)(closest_idx);
        
    case 'closest_to_center'
        % Use the detection closest to window center
        window_center = (window_start + window_end) / 2;
        [~, closest_idx] = min(abs(window_detections.concat_ts - window_center));
        window_groups = window_detections.(groups_column)(closest_idx);
        
    case 'majority'
        % Use the most common group structure
        all_group_structures = window_detections.(groups_column);
        
        if isempty(all_group_structures)
            window_groups = {};
            return;
        end
        
        % Convert group structures to strings for comparison
        group_strings = cell(length(all_group_structures), 1);
        for j = 1:length(all_group_structures)
            group_struct = all_group_structures{j};
            if isempty(group_struct)
                group_strings{j} = '[]';
            else
                % Convert each group to a sorted string representation
                group_str_parts = cell(length(group_struct), 1);
                                 for k = 1:length(group_struct)
                     group = group_struct{k};
                     if isempty(group)
                         group_str_parts{k} = '[]';
                     else
                         % Convert column vector to row vector for sorting and string conversion
                         group_row = group(:)';  % Ensure it's a row vector
                         group_str_parts{k} = ['[', num2str(sort(group_row)), ']'];
                     end
                 end
                % Sort the groups and join them
                group_strings{j} = strjoin(sort(group_str_parts), ',');
            end
        end
        
        % Find the most common group structure
        unique_structures = unique(group_strings);
        max_count = 0;
        majority_idx = 1; % Default to first if no clear majority
        
        for j = 1:length(unique_structures)
            count = sum(strcmp(group_strings, unique_structures{j}));
            if count > max_count
                max_count = count;
                majority_idx = j;
            end
        end
        
        % Get the first occurrence of the majority structure
        majority_string = unique_structures{majority_idx};
        majority_indices = find(strcmp(group_strings, majority_string));
        window_groups = all_group_structures{majority_indices(1)};
        
    case 'union'
        % Combine all groups from the window period
        all_groups = {};
        for j = 1:height(window_detections)
            all_groups = [all_groups; window_detections.(groups_column){j}];
        end
        % Remove duplicates while preserving order
        if ~isempty(all_groups)
            unique_groups = {};
            for j = 1:length(all_groups)
                if ~any(cellfun(@(x) isequal(x, all_groups{j}), unique_groups))
                    unique_groups{end+1} = all_groups{j};
                end
            end
            window_groups = unique_groups;
        else
            window_groups = {};
        end
        
    otherwise
        error('Unknown aggregation method: %s', method);
end

% window_groups = window_groups{1};

end

function filtered_speakers = filterSpeakersByGroups(continuous_speakers, window_groups)
% Filter continuous speakers to only include those who appear in any of the detected groups
%
% Inputs:
%   continuous_speakers - Array of person IDs who spoke continuously
%   window_groups - Cell array of groups, each group is an array of person IDs
%
% Output:
%   filtered_speakers - Array of person IDs who both spoke continuously AND appear in groups

filtered_speakers = [];

% Handle case where window_groups is empty
if isempty(window_groups)
    return; % No groups detected, so no filtered speakers
end

% Get all people who appear in any group
all_group_members = [];
for i = 1:length(window_groups)
    group = window_groups{i};
    all_group_members = [all_group_members; group];
end

% Filter continuous speakers to only those who appear in groups
for speaker_id = continuous_speakers
    if ismember(speaker_id, all_group_members)
        filtered_speakers = [filtered_speakers, speaker_id];
    end
end

end

function total_groups = countTotalGroups(filtered_speakers, window_groups)
% Count unique groups that filtered speakers belong to
%
% Inputs:
%   filtered_speakers - Array of person IDs who both spoke continuously AND appear in groups
%   window_groups - Cell array of groups, each group is an array of person IDs
%
% Output:
%   total_groups - Number of unique groups that filtered speakers belong to

% Handle case where window_groups is empty
if isempty(window_groups)
    total_groups = 0;
    return;
end

% Find which groups contain any of the filtered speakers
groups_with_speakers = [];
for i = 1:length(window_groups)
    group = window_groups{i};
    % Check if any filtered speaker belongs to this group
    if any(ismember(filtered_speakers, group))
        groups_with_speakers = [groups_with_speakers, i];
    end
end

% Count unique groups
total_groups = length(groups_with_speakers);

end
