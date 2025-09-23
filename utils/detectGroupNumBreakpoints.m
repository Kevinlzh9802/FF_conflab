%% Detect how many groups simultaneous speakers belong to using breakpoint-based windows
% This script analyzes group detection results to find breakpoints where
% group structures change, then uses these breakpoints as window boundaries
% instead of rolling windows

% Load required data
load('../data/speaking_status.mat', 'sp_merged');

% Parameters
clues = {'headRes', 'shoulderRes', 'hipRes', 'footRes'};
cameras = [2, 4, 6, 8];
videos = [2, 3];

% Initialize results storage
window_table = table();

% Process each video and camera combination separately
for vid = videos
    fprintf('Processing video %d...\n', vid);
    
    for cam = cameras
        fprintf('  Processing camera %d...\n', cam);
        
        % Find breakpoints for this specific video and camera combination
        breakpoints = findBreakpointsForCamera(data_results, clues, vid, cam);
        
        if isempty(breakpoints)
            fprintf('    No breakpoints found for video %d, camera %d\n', vid, cam);
            continue;
        end
        
        % Generate windows based on breakpoints
        windows = generateBreakpointWindows(breakpoints, vid, cam);
        
        % Process each window
        for w = 1:length(windows)
            window = windows{w};
            fprintf('    Processing window %d: frames %d-%d\n', w, window.start, window.end);
            
            % Get speaking status for this window
            speaking_status_window = getSpeakingStatusForWindow(sp_merged{vid}, window.start, window.end);
            
            % Create row for table
            row = table();
            row.id = w;
            row.Vid = vid;
            row.Cam = cam;
            row.time = {[window.start, window.end]};
            row.length = window.end - window.start + 1;
            row.speaking_all_time = {speaking_status_window.speaking_all_time};
            
            % Append to table
            window_table = [window_table; row];
        end
    end
end

% Run countSpeakerGroups with closest_to_start aggregation
fprintf('Running countSpeakerGroups with breakpoint-based windows...\n');
window_table = countSpeakerGroups(window_table, data_results, clues, 'closest_to_start');

% Process the results
fprintf('Processing window table results...\n');
[filtered_table, pairwise_diffs] = processWindowTable(window_table, 'closest_to_start');

% Display summary
fprintf('\n=== Breakpoint-based Analysis Summary ===\n');
fprintf('Total windows created: %d\n', height(window_table));
fprintf('Valid windows (after filtering): %d\n', height(filtered_table));
fprintf('Pairwise differences matrix:\n');
disp(pairwise_diffs);

%% Functions

function breakpoints = findBreakpointsForCamera(data_results, clues, vid, cam)
% FINDBREAKPOINTSFORCAMERA - Find all timestamps where group detection results change
% across all clues for a specific video and camera combination
%
% Inputs:
%   data_results - Table with group detection results
%   clues - Cell array of clue column names (e.g., {'headRes', 'shoulderRes', ...})
%   vid - Video ID to filter on
%   cam - Camera ID to filter on
%
% Output:
%   breakpoints - Sorted array of unique timestamps where group structures change

% Filter data for this video and camera
camera_data = data_results(data_results.Vid == vid & data_results.Cam == cam, :);
if isempty(camera_data)
    breakpoints = [];
    return;
end

% Get all timestamps for this camera
all_timestamps = unique(camera_data.concat_ts);
all_timestamps = all_timestamps(~isnan(all_timestamps));
all_timestamps = sort(all_timestamps);

if length(all_timestamps) <= 1
    breakpoints = all_timestamps;
    return;
end

% Find breakpoints by comparing consecutive timestamps
breakpoints = [];
prev_groups = cell(length(clues), 1);

% Initialize previous groups for first timestamp
for t_idx = 1:length(all_timestamps)
    current_timestamp = all_timestamps(t_idx);
    
    % Get current groups for all clues
    current_groups = cell(length(clues), 1);
    has_changes = false;
    
    for c_idx = 1:length(clues)
        clue = clues{c_idx};
        
        % Get group detection result for this timestamp and clue
        row_idx = find(camera_data.concat_ts == current_timestamp, 1);
        if ~isempty(row_idx)
            current_groups{c_idx} = camera_data.(clue){row_idx};
        else
            current_groups{c_idx} = {};
        end
        
        % Compare with previous groups (if not first timestamp)
        if t_idx > 1
            if ~isequal(current_groups{c_idx}, prev_groups{c_idx})
                has_changes = true;
            end
        end
    end
    
    % If changes detected, add this timestamp as a breakpoint
    if has_changes || t_idx == 1
        breakpoints = [breakpoints, current_timestamp];
    end
    
    % Update previous groups for next iteration
    prev_groups = current_groups;
end

% Ensure we have at least the first and last timestamps
if isempty(breakpoints) || breakpoints(1) ~= all_timestamps(1)
    breakpoints = [all_timestamps(1), breakpoints];
end
if breakpoints(end) ~= all_timestamps(end)
    breakpoints = [breakpoints, all_timestamps(end)];
end

% Remove duplicates and sort
breakpoints = unique(breakpoints);
breakpoints = sort(breakpoints);

fprintf('    Found %d breakpoints for video %d, camera %d\n', length(breakpoints), vid, cam);
end

function windows = generateBreakpointWindows(breakpoints, vid, cam)
% GENERATEBREAKPOINTWINDOWS - Generate windows based on breakpoints
%
% Inputs:
%   breakpoints - Array of timestamps where group structures change
%   vid - Video ID (for logging purposes)
%   cam - Camera ID (for logging purposes)
%
% Output:
%   windows - Cell array of window structures with start and end times

windows = {};

if length(breakpoints) < 2
    warning('Not enough breakpoints to create windows for video %d, camera %d', vid, cam);
    return;
end

% Create windows between consecutive breakpoints
for i = 1:length(breakpoints)-1
    window = struct();
    window.start = breakpoints(i);
    window.end = breakpoints(i+1);
    windows{end+1} = window;
end

fprintf('    Generated %d windows for video %d, camera %d\n', length(windows), vid, cam);
end

function speaking_status_window = getSpeakingStatusForWindow(actions, window_start, window_end)
% GETSPEAKINGSTATUSFORWINDOW - Extract speaking status for a specific window
%
% Inputs:
%   actions - Matrix where first row contains participant IDs, 
%            subsequent rows contain speaking status (0/1) for each time frame
%   window_start - Start frame of the window
%   window_end - End frame of the window
%
% Output:
%   speaking_status_window - Structure containing window information and continuous speakers

participant_ids = actions(1, :);
speaking_data = actions(2:end, :);

% Ensure window bounds are within data range
window_start = max(1, window_start);
window_end = min(size(speaking_data, 1), window_end);

% Extract window data
window_speaking_data = speaking_data(window_start:window_end, :);

% Create a structure to store speaking status
speaking_status_window = struct();

% Window length
speaking_status_window.window_length = window_end - window_start + 1;

% Start and end time (in frame indices)
speaking_status_window.start_time = window_start;
speaking_status_window.end_time = window_end;

% Find participants who spoke for ALL time during the window
continuous_speakers = [];

for p_idx = 1:length(participant_ids)
    participant_speaking = window_speaking_data(:, p_idx);
    % Check if this participant spoke in ALL frames (all values are 1)
    if all(participant_speaking == 1)
        continuous_speakers = [continuous_speakers, participant_ids(p_idx)];
    end
end

speaking_status_window.speaking_all_time = continuous_speakers;
speaking_status_window.participant_ids = participant_ids;
end
