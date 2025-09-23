%% Test script for breakpoint-based group detection
% This script demonstrates how the new breakpoint-based approach works
% by showing the difference between rolling windows and breakpoint windows

% Load required data
load('../data/data_results.mat', 'data_results');
load('../data/speaking_status.mat', 'sp_merged');

% Test parameters
test_vid = 2;
test_cam = 2;
clues = {'headRes', 'shoulderRes', 'hipRes', 'footRes'};

fprintf('=== Testing Breakpoint Detection ===\n');
fprintf('Video: %d, Camera: %d\n', test_vid, test_cam);

% Filter data for test video and camera
test_data = data_results(data_results.Vid == test_vid & data_results.Cam == test_cam, :);
test_data = sortrows(test_data, 'concat_ts');

if isempty(test_data)
    fprintf('No data found for video %d, camera %d\n', test_vid, test_cam);
    return;
end

fprintf('Found %d detection timestamps\n', height(test_data));

% Show first few timestamps and their group detections
fprintf('\nFirst 10 timestamps and their group detections:\n');
fprintf('Timestamp | Head Groups | Shoulder Groups | Hip Groups | Foot Groups\n');
fprintf('----------|-------------|-----------------|------------|------------\n');

for i = 1:min(10, height(test_data))
    row = test_data(i, :);
    timestamp = row.concat_ts;
    
    % Format group detections for display
    head_groups = formatGroupsForDisplay(row.headRes{1});
    shoulder_groups = formatGroupsForDisplay(row.shoulderRes{1});
    hip_groups = formatGroupsForDisplay(row.hipRes{1});
    foot_groups = formatGroupsForDisplay(row.footRes{1});
    
    fprintf('%8d | %-11s | %-15s | %-10s | %-10s\n', ...
        timestamp, head_groups, shoulder_groups, hip_groups, foot_groups);
end

% Find breakpoints for this specific camera
fprintf('\n=== Finding Breakpoints ===\n');
breakpoints = findBreakpointsForCamera(test_data, clues);

fprintf('Breakpoints found: %s\n', mat2str(breakpoints));

% Generate windows
fprintf('\n=== Generated Windows ===\n');
windows = generateBreakpointWindows(breakpoints, test_vid, test_cam);

for i = 1:length(windows)
    window = windows{i};
    fprintf('Window %d: frames %d-%d (length: %d)\n', ...
        i, window.start, window.end, window.end - window.start + 1);
end

% Compare with rolling window approach
fprintf('\n=== Comparison with Rolling Windows ===\n');
window_size = 60;  % 1 second at 60fps
step_size = 60;

% Calculate how many rolling windows would be created
total_frames = max(test_data.concat_ts) - min(test_data.concat_ts) + 1;
num_rolling_windows = floor((total_frames - window_size) / step_size) + 1;

fprintf('Rolling windows (size %d, step %d): %d windows\n', ...
    window_size, step_size, num_rolling_windows);
fprintf('Breakpoint windows: %d windows\n', length(windows));

if length(windows) < num_rolling_windows
    fprintf('Breakpoint approach creates %.1f%% fewer windows\n', ...
        100 * (1 - length(windows) / num_rolling_windows));
else
    fprintf('Breakpoint approach creates %.1f%% more windows\n', ...
        100 * (length(windows) / num_rolling_windows - 1));
end

%% Helper Functions

function breakpoints = findBreakpointsForCamera(data, clues)
% Find breakpoints for a specific camera's data

if height(data) <= 1
    breakpoints = data.concat_ts;
    return;
end

timestamps = data.concat_ts;
breakpoints = [];
prev_groups = cell(length(clues), 1);

% Initialize previous groups for first timestamp
for t_idx = 1:length(timestamps)
    current_timestamp = timestamps(t_idx);
    row_idx = find(data.concat_ts == current_timestamp, 1);
    
    % Get current groups for all clues
    current_groups = cell(length(clues), 1);
    has_changes = false;
    
    for c_idx = 1:length(clues)
        clue = clues{c_idx};
        current_groups{c_idx} = data.(clue){row_idx};
        
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
if isempty(breakpoints) || breakpoints(1) ~= timestamps(1)
    breakpoints = [timestamps(1), breakpoints];
end
if breakpoints(end) ~= timestamps(end)
    breakpoints = [breakpoints, timestamps(end)];
end

breakpoints = unique(breakpoints);
breakpoints = sort(breakpoints);
end

function windows = generateBreakpointWindows(breakpoints, vid, cam)
% Generate windows based on breakpoints

windows = {};

if length(breakpoints) < 2
    return;
end

% Create windows between consecutive breakpoints
for i = 1:length(breakpoints)-1
    window = struct();
    window.start = breakpoints(i);
    window.end = breakpoints(i+1);
    windows{end+1} = window;
end
end

function group_str = formatGroupsForDisplay(groups)
% Format group detection results for display

if isempty(groups)
    group_str = '[]';
    return;
end

group_parts = cell(length(groups), 1);
for i = 1:length(groups)
    group = groups{i};
    if isempty(group)
        group_parts{i} = '[]';
    else
        group_parts{i} = ['[', num2str(group(:)'), ']'];
    end
end

group_str = strjoin(group_parts, ',');
if length(group_str) > 10
    group_str = [group_str(1:7), '...'];
end
end
