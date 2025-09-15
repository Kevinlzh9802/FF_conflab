close all;

% Run plotting for each used_part directly
for part_idx = 1:length(params.used_parts)
    part_str = char(params.used_parts(part_idx));
    
    % Extract cam, vid, seg from the string
    if length(part_str) >= 3
        cam = str2double(part_str(1));
        vid = str2double(part_str(2));
        seg = str2double(part_str(3));
        
        % Plot for each clue
        for clue = clues
            plotSingleSegment(clue, cam, vid, seg, results, use_real, speaking_status, frames);
        end
    end
end

%% Function to format cell array into string
function outputStr = formatCellArray(cellArray)
% FORMATCELLARRAY Converts a nested cell array into a formatted string.
%
% Example:
%   cellArray = {{[32 15 2 17]}, {[22 11]}, {[25 10]}, {[3]}};
%   outputStr = formatCellArray(cellArray);
%
% Output:
%   '{{32,15,2,17}}, {{22,11}}, {{25,10}}, {{3}}'

    % Convert each numeric array inside the cell to a string with commas
    formattedCells = cellfun(@(x) mat2str(x, 2), cellArray, 'UniformOutput', false);

    % Wrap each formatted cell with curly braces
    outputStr = strjoin(formattedCells, ', ');
end

function plotGroupPolygon(ax, data, groups, GTgroups, scale)
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
text(ax, 0.5, -0.1, ['GT: ', formatCellArray(GTgroups)], 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'FontSize', 18);

hold(ax, "off");
end


function plotSingleSegment(clue, cam, vid, seg, results, use_real, speaking_status, frames)
% PLOTSINGLESEGMENT Plot groups for a single segment
%
% Inputs:
%   clue - body part clue (e.g., 'head', 'shoulder', etc.)
%   cam - camera ID
%   vid - video ID
%   seg - segment ID
%   results - results structure containing group data
%   use_real - boolean for real vs image coordinates
%   speaking_status - speaking status data
%   frames - frame data

f_name = clue + "Res";
feat_name = clue + "Feat";

% Create folder name for this specific cam-vid-seg combination
seg_folder_name = "cam" + cam + "_vid" + vid + "_seg" + seg + "_" + clue;
if use_real
    folder_path = "../data/results/" + seg_folder_name + "_real/";
else
    folder_path = "../data/results/" + seg_folder_name + "/";
end
mkdir(folder_path);

% Get data for this specific cam-vid-seg combination
used_data = results.(clue).original_data;
% Filter data for this specific cam-vid-seg
mask = (used_data.Cam == cam) & (used_data.Vid == vid) & (used_data.Seg == seg);
used_data = used_data(mask, :);

if height(used_data) == 0
    fprintf('No data found for cam%d_vid%d_seg%d_%s\n', cam, vid, seg, clue);
    return;
end

features = used_data.(feat_name);
hfig = figure('Units','pixels','Position',[100 100 960 540]); % Fixed size
ax = axes(hfig);

for f=1:height(used_data)
    img = findMatchingFrame(used_data, frames, f);
    
    f_info = table2struct(used_data(f, {'Cam', 'Vid', 'Seg', 'Timestamp'}));

    [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, f_info.Vid, ...
        f_info.Seg, 1, 1);
    [speaking, confidence] = readSpeakingStatus(speaking_status, f_info.Vid, ...
        f_info.Seg, f_info.Timestamp+1, 1);
    
    GTgroups = used_data.GT{f};
    groups = used_data.(f_name){f};

    if use_real
        scale=1;
        interval = 25:28;
        feat_plot = features{f}(:, [25, 29:48]);
    else
        scale=0.5;
        interval = 1:4;
        feat_plot = features{f}(:, [1, 5:24]);
    end
    plotSkeletonOnImage(ax, img, feat_plot, [1,2,3,4,5], use_real);
    plotGroupPolygon(ax, features{f}(:, interval), groups, GTgroups, scale);

    % Write frame
    frame = getframe(hfig);
    imwrite(frame.cdata, folder_path + "frame" + num2str(f) + ".png");
end

close(hfig);
fprintf('Completed plotting for cam%d_vid%d_seg%d_%s\n', cam, vid, seg, clue);

end