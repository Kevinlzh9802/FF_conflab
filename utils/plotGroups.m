close all;

clues = ["head", "shoulder", "hip", "foot"];
for clue_id=[1,2,3,4]
% clue_id = 1;
clue = clues(clue_id);
f_name = clue + "Res";
feat_name = clue + "Feat";

seg_folder_name = "cam" + params.cams + "_vid" + params.vids +...
    "_seg" + params.segs + "_" + clue;
if use_real
    folder_path = "../data/results/" + seg_folder_name + "_real/";
else
    folder_path = "../data/results/" + seg_folder_name + "/";
end
mkdir(folder_path);

% load('../data/frames.mat', 'frames');

% output_video = clue + "_cam4_vid2_seg8.avi";
frame_rate = 10;
% v = VideoWriter(output_video);
% v.FrameRate = frame_rate;
% open(v);
used_data = results.(clue).original_data;
features = used_data.(feat_name);
hfig = figure('Units','pixels','Position',[100 100 960 540]); % Fixed size
% hfig2 = figure('Units','pixels','Position',[100 100 960 540]); % Fixed size
ax = axes(hfig);
% ax2 = axes(hfig2);
for f=1:height(used_data)

    img = findMatchingFrame(used_data, frames, f);
    % Show image
    % imshow(img); hold on;

    f_info = table2struct(used_data(f, {'Cam', 'Vid', 'Seg', 'Timestamp'}));

    [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, f_info.Vid, ...
        f_info.Seg, 1, 1);
    [speaking, confidence] = readSpeakingStatus(speaking_status, f_info.Vid, ...
        f_info.Seg, f_info.Timestamp+1, 1);
    
    GTgroups = used_data.GT{f};
    % groups = used_data.(f_name){f};
    
    % disp_info = struct();
    % disp_info.GT = GTgroups;
    % disp_info.detection = groups;
    % disp_info.speaking = getStatusForGroup(sp_ids, speaking, GTgroups);
    % disp_info.confidence = getStatusForGroup(cf_ids, confidence, GTgroups);
    % disp_info.kp = readPoseInfo(f_info, features{f}(:,1));

    % plotFrustumsWithImage(features{f}, params.frustum, img, disp_info, [4]);
    % feat_pixel = features{f}(:, [1, 5:20]);
    % feat_real = features{f}(:, [21, 25:40]);

    % plotSkeletonOnImage(ax1, img, feat_pixel, [1,2,3,4], false);
    
    groups = used_data.(f_name){f};

    if use_real
        scale=1;
        interval = 21:24;
        feat_plot = features{f}(:, [21, 25:40]);
    else
        scale=0.5;
        interval = 1:4;
        feat_plot = features{f}(:, [1, 5:20]);
    end
    plotSkeletonOnImage(ax, img, feat_plot, [1,2,3,4], use_real);
    plotGroupPolygon(ax, features{f}(:, interval), groups, GTgroups, scale);

    % Write frame
    frame = getframe(hfig);
    imwrite(frame.cdata, folder_path + "frame" + num2str(f) + ".png");
    % c = 9;
% close(v);
% disp('Video saved.');
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