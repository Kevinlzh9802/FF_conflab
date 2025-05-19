close all;
clue = "head";
f_name = clue + "Res";
feat_name = clue + "Feat";
features = results.(clue).groups;
% load('../data/frames.mat', 'frames');

output_video = clue + "_cam4_vid2_seg8.avi";
frame_rate = 10;
v = VideoWriter(output_video);
v.FrameRate = frame_rate;
open(v);
used_data = results.(clue).original_data;

hfig = figure('Units','pixels','Position',[100 100 960 540]); % Fixed size
for f=1:height(used_data)

    img = findMatchingFrame(used_data, frames, f);
    % Show image
    imshow(img); hold on;

    % Plot each person
    data = used_data.(feat_name){f};
    for i = 1:size(data, 1)
        x = data(i, 2) * 0.5;
        y = data(i, 3) * 0.5;
        theta = data(i, 4);

        % Plot position
        plot(x, y, 'ro', 'MarkerFaceColor', 'r');

        % Plot orientation as arrow (length = 20 pixels)
        len = 20;
        u = len * cos(theta);
        v_arrow = len * sin(theta);
        quiver(x, y, u, v_arrow, 0, 'Color', 'b', 'LineWidth', 1.5, 'MaxHeadSize', 2);
    end

    % Plot groups
    groups = used_data.(f_name){f};
    for g = 1:length(groups)
        group_ids = groups{g};
        idx = ismember(data(:,1), group_ids);
        group_data = data(idx, 2:3) * 0.5;  % positions

        if size(group_data,1) >= 3
            % Plot convex hull
            k = convhull(group_data(:,1), group_data(:,2));
            plot(group_data(k,1), group_data(k,2), 'g-', 'LineWidth', 2);
        elseif size(group_data,1) == 2
            % Plot line
            plot(group_data(:,1), group_data(:,2), 'g-', 'LineWidth', 2);
        elseif size(group_data,1) == 1
            % Plot a small circle around it
            viscircles(group_data, 15, 'Color', 'g', 'LineWidth', 1);
        end
    end

    % Write frame
    frame = getframe(hfig);
    writeVideo(v, frame);

    hold off;
    c = 9;
end
close(v);
disp('Video saved.');