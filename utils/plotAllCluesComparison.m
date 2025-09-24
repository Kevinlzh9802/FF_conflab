% Plot comparison of all clues (head, shoulder, hip, foot) on the same figure
% Based on plotGroups.m structure - shows detected formations as polygons
% This script can be called directly

% Load required data (assuming these variables exist in workspace)
% data_results, frames, speaking_status, use_real should be available

% Create figure with 2x3 subplot layout
figure('Name', 'Formation Detection - All Clues Comparison', 'Position', [100, 100, 1800, 600]);

% Define clue types and their subplot positions
clues = ["head", "shoulder", "hip", "foot"];
subplot_positions = [1, 2, 4, 5]; % Positions 1,2,4,5 in 2x3 grid

% Set default values if variables don't exist
if ~exist('data_results', 'var')
    error('data_results variable not found in workspace. Please load the data first.');
end
if ~exist('frames', 'var')
    error('frames variable not found in workspace. Please load the data first.');
end
if ~exist('speaking_status', 'var')
    error('speaking_status variable not found in workspace. Please load the data first.');
end
if ~exist('use_real', 'var')
    use_real = false; % Default to image coordinates
end
if ~exist('outdir', 'var')
    outdir = '../data/results/'; % Default output directory
end

% Get all unique cam-vid-seg combinations
if height(data_results) > 0
    unique_combinations = unique(data_results(:, {'Cam', 'Vid', 'Seg'}), 'rows');
    
    % Process each cam-vid-seg combination
    for combo_idx = 1:height(unique_combinations)
        cam = unique_combinations.Cam(combo_idx);
        vid = unique_combinations.Vid(combo_idx);
        seg = unique_combinations.Seg(combo_idx);
        
        fprintf('Processing cam%d_vid%d_seg%d (%d/%d)\n', cam, vid, seg, combo_idx, height(unique_combinations));
        
        % Create subfolder for this combination
        combo_folder = sprintf('cam%d_vid%d_seg%d', cam, vid, seg);
        combo_path = fullfile(outdir, combo_folder);
        if ~exist(combo_path, 'dir')
            mkdir(combo_path);
        end
        
        % Get data for this specific cam-vid-seg combination
        mask = (data_results.Cam == cam) & (data_results.Vid == vid) & (data_results.Seg == seg);
        used_data = data_results(mask, :);
        
        if height(used_data) > 0
            % Get all frames for this combination
            unique_frames = unique(used_data.Timestamp);
            
            % Process each frame
            for frame_idx = 1:length(unique_frames)
                timestamp = unique_frames(frame_idx);
                frame_mask = used_data.Timestamp == timestamp;
                frame_data = used_data(frame_mask, :);
                
                % Create figure for this frame
                figure('Name', sprintf('Frame %d - cam%d_vid%d_seg%d', timestamp, cam, vid, seg), ...
                    'Position', [100, 100, 1800, 1000], 'Visible', 'off');
                
                % Plot each clue for this frame
                for clue_idx = 1:length(clues)
                    clue = clues(clue_idx);
                    subplot_pos = subplot_positions(clue_idx);
                    
                    % Get data for this specific clue and frame
                    clue_mask = frame_data.Cam == cam & frame_data.Vid == vid & frame_data.Seg == seg;
                    clue_data = frame_data(clue_mask, :);
                    
                    if height(clue_data) > 0
                        % Plot the detected formations for this clue
                        plotClueFormations(subplot_pos, clue, clue_data, frames, use_real, timestamp);
                    else
                        % Plot placeholder if no data
                        subplot(2, 3, subplot_pos);
                        text(0.5, 0.5, sprintf('No data for %s', clue), 'HorizontalAlignment', 'center', ...
                            'VerticalAlignment', 'middle', 'FontSize', 14);
                        title(clue);
                        axis off;
                    end
                end
                
                % Plot original frame in position 6 (bottom right)
                subplot(2, 3, 6);
                % Get GT data from the first clue (assuming all clues have same GT for same frame)
                if height(frame_data) > 0
                    first_clue_data = frame_data(1, :);
                    GTgroups = first_clue_data.GT{1};
                else
                    GTgroups = {};
                end
                plotOriginalFrame(cam, vid, seg, frames, timestamp, GTgroups);
                
                % Save the figure
                frame_filename = sprintf('frame_%d_comparison.png', timestamp);
                saveas(gcf, fullfile(combo_path, frame_filename));
                close(gcf);
            end
            
            fprintf('Saved %d frames for cam%d_vid%d_seg%d\n', length(unique_frames), cam, vid, seg);
        else
            fprintf('No data found for cam%d_vid%d_seg%d\n', cam, vid, seg);
        end
    end
    
    fprintf('Completed processing all cam-vid-seg combinations\n');
    
else
    fprintf('No data available in data_results\n');
end

function plotClueFormations(subplot_pos, clue, used_data, frames, use_real, timestamp)
% Plot detected formations for a specific clue
% Based on plotSingleSegment structure from plotGroups.m

f_name = clue + "Res";
feat_name = clue + "Feat";

% Get the frame data for the specific timestamp
f = 1; % Use first row since we filtered for specific timestamp
img = findMatchingFrame(used_data, frames, f);

groups = used_data.(f_name){f};
features = used_data.(feat_name);

if use_real
    scale = 1;
    interval = 25:28;
    feat_plot = features{f}(:, [25, 29:48]);
else
    scale = 0.5;
    interval = 1:4;
    feat_plot = features{f}(:, [1, 5:24]);
end

% Create subplot
subplot(2, 3, subplot_pos);
ax = gca;

% Plot skeleton on image
plotSkeletonOnImage(ax, img, feat_plot, [1,2,3,4,5], use_real);
plotGroupPolygon(ax, features{f}(:, interval), groups, scale);

% Add text showing detection results (groups)
if ~isempty(groups)
    groups_text = formatCellArray(groups);
    text(ax, 0.5, -0.15, ['Detected: ', groups_text], 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'blue');
end

title(sprintf('%s (t=%d)', clue, timestamp));
end

function plotOriginalFrame(cam, vid, seg, frames, timestamp, GTgroups)
% Plot the original frame without any group overlays
% Get the actual image from frames table and show GT information

try
    % Find the matching row in frames table
    frame_mask = (frames.Cam == cam) & (frames.Vid == vid) & ...
        (frames.Seg == seg) & (cell2mat(frames.Timestamp) == timestamp);
    frame_row = frames(frame_mask, :);
    
    if height(frame_row) > 0
        % Get the image from FrameData column
        img = frame_row.FrameData{1};
        
        % Display the image
        imshow(img);
        
        % Add GT information text
        if ~isempty(GTgroups)
            gt_text = formatCellArray(GTgroups);
            text(0.5, -0.1, ['GT: ', gt_text], 'Units', 'normalized', ...
                'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', 'red', 'FontWeight', 'bold');
        end
        
        title(sprintf('Original Frame (t=%d)', timestamp));
    else
        % No matching frame found
        text(0.5, 0.5, sprintf('No frame found\nCam%d Vid%d Seg%d\nt=%d', cam, vid, seg, timestamp), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
        title(sprintf('Original Frame (t=%d)', timestamp));
        axis off;
    end
catch ME
    % Error occurred, show error message
    text(0.5, 0.5, sprintf('Error loading frame\nCam%d Vid%d Seg%d\nt=%d\n%s', cam, vid, seg, timestamp, ME.message), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12);
    title(sprintf('Original Frame (t=%d)', timestamp));
    axis off;
end
end

function plotGroupPolygon(ax, data, groups, scale)
% Plot each person
for i = 1:size(data, 1)
    x = data(i, 2) * scale;
    y = data(i, 3) * scale;
    theta = data(i, 4);

    % Plot position
    % plot(x, y, 'ro', 'MarkerFaceColor', 'r');

    % Plot orientation as arrow (length = 20 pixels)
    len = 20;
    u = len * cos(theta);
    v_arrow = len * sin(theta);
    hold(ax, "on");
    quiver(ax, x, y, u, v_arrow, 0, 'Color', 'b', 'LineWidth', 1.5, 'MaxHeadSize', 2);
end

% Plot groups
% groups = used_data.(f_name){f};
for g = 1:length(groups)
    group_ids = groups{g};
    idx = ismember(data(:,1), group_ids);
    group_data = data(idx, 2:3) * scale;  % positions

    if size(group_data,1) >= 3
        % Plot convex hull
        try
            k = convhull(group_data(:,1), group_data(:,2));
        catch
            continue;
        end

        plot(ax, group_data(k,1), group_data(k,2), 'g-', 'LineWidth', 2);
    elseif size(group_data,1) == 2
        % Plot line
        plot(ax, group_data(:,1), group_data(:,2), 'g-', 'LineWidth', 2);
    elseif size(group_data,1) == 1
        % Plot a small circle around it
        viscircles(ax, group_data, 15, 'Color', 'g', 'LineWidth', 1);
    end
end

% Text GT groups
text(ax, 0.5, -0.1, ['Detection: ', formatCellArray(groups)], 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'FontSize', 18);

hold(ax, "off");
end

function outputStr = formatCellArray(cellArray)
% FORMATCELLARRAY Converts a nested cell array into a formatted string.
% Handles various formats: empty, n*1 or 1*n cell arrays, row/column vectors
%
% Examples:
%   cellArray = {{[32 15 2 17]}, {[22 11]}, {[25 10]}, {[3]}};
%   cellArray = {[32 15 2 17], [22 11], [25 10], [3]};
%   cellArray = {[32; 15; 2; 17], [22; 11]};
%   cellArray = {};

    % Handle empty cell array
    if isempty(cellArray)
        outputStr = '{}';
        return;
    end
    
    % Ensure cellArray is a column vector for consistent processing
    if isrow(cellArray)
        cellArray = cellArray';
    end
    
    % Process each element
    formattedCells = cell(length(cellArray), 1);
    for i = 1:length(cellArray)
        element = cellArray{i};
        
        % Handle different element types
        if isempty(element)
            formattedCells{i} = '[]';
        elseif iscell(element)
            % Nested cell - recurse
            formattedCells{i} = ['{', formatCellArray(element), '}'];
        elseif isnumeric(element)
            % Numeric array - convert to string
            % Ensure it's a row vector for consistent display
            if iscolumn(element)
                element = element';
            end
            formattedCells{i} = mat2str(element, 2);
        else
            % Other types - convert to string
            formattedCells{i} = char(string(element));
        end
    end
    
    % Join all formatted elements
    outputStr = strjoin(formattedCells, ', ');
end