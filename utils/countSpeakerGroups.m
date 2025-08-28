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

% Add total_groups column to window_table
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
        window_table.total_groups(i) = 0; % No continuous speakers, so 0 groups
        continue;
    end
    
    % Get group detection results for this window period
    window_groups = getGroupsForWindow(group_detection_table, groups_column, window_start, window_end, vid, aggregation_method);
    
    % Count total groups for all continuous speakers in this window
    total_groups = 0;
    for speaker_idx = 1:length(continuous_speakers)
        speaker_id = continuous_speakers(speaker_idx);
        num_groups = countGroupsForSpeaker(speaker_id, window_groups);
        total_groups = total_groups + num_groups;
    end
    
    % Store total groups in the window_table
    window_table.total_groups(i) = total_groups;
end

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
        % Use the most common group structure (simplified implementation)
        % For now, just use the first detection (can be enhanced later)
        window_groups = window_detections.(groups_column)(1);
        
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

window_groups = window_groups{1};

end

function num_groups = countGroupsForSpeaker(speaker_id, window_groups)
% Count how many groups a speaker belongs to in the given group structure
%
% Inputs:
%   speaker_id - ID of the speaker
%   window_groups - Cell array of groups, each group is an array of person IDs
%                   Can be empty [] or contain non-empty groups like {[1,2,3],[4,5]}
%
% Output:
%   num_groups - Number of groups the speaker belongs to

num_groups = 0;

% Handle case where window_groups is empty
if isempty(window_groups)
    return; % No groups detected, speaker belongs to 0 groups
end

for i = 1:length(window_groups)
    group = window_groups{i};
    if ismember(speaker_id, group)
        num_groups = num_groups + 1;
    end
end

end
