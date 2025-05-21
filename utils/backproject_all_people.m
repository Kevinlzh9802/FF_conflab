
% Example visualization script
% Define camera parameters K, R, t beforehand
% Assume frame_data and height_ratios_map are available

worldXY_all = backProject(frame_data, height_ratios_map, 1.7, K, R, t);

figure;
hold on;
numPeople = size(worldXY_all, 1);
numParts = size(worldXY_all, 3);

colors = lines(numParts);
labels = fieldnames(height_ratios_map);

for i = 1:numPeople
    for j = 1:numParts
        plot(worldXY_all(i,1,j), worldXY_all(i,2,j), 'o', 'Color', colors(j,:), 'MarkerSize', 8);
    end
end

legend(labels, 'Interpreter', 'none');
xlabel('World X');
ylabel('World Y');
title('Backprojected Keypoints on Ground Plane');
grid on;
axis equal;

% Function to backproject pixel points onto a horizontal plane at specified height
function worldXY_all = backProject(frame_data, height_ratios_map, bodyHeight, K, R, t)
% frame_data: Nx20 matrix where each row is a person with keypoint pixel data
% height_ratios_map: struct with field names corresponding to body parts, each with height ratio (0=head, 1=foot)
% bodyHeight: assumed person height in world coordinates
% K, R, t: camera intrinsic and extrinsic parameters

    numPeople = size(frame_data, 1);
    body_parts = fieldnames(height_ratios_map);
    numParts = numel(body_parts);

    worldXY_all = zeros(numPeople, 2, numParts); % [X, Y] for each person and part

    Kinv = inv(K);
    Rinv = R';
    camCenter = -Rinv * t;

    % Mapping from frame_data columns to part names
    part_column_map = struct('head', [5,6], 'leftFoot', [17,18], 'rightFoot', [19,20], ...
                             'leftHip', [13,14], 'rightHip', [15,16], 'leftShoulder', [9,10], 'rightShoulder', [11,12]);

    for p = 1:numPeople
        partIdx = 1;
        for part = body_parts'
            name = part{1};
            cols = part_column_map.(name);
            uv = [frame_data(p, cols), 1]';  % [u; v; 1]
            dir_cam = Kinv * uv;
            dir_world = Rinv * dir_cam;

            Z = height_ratios_map.(name) * bodyHeight;
            lambda = (Z - camCenter(3)) / dir_world(3);
            world_point = camCenter + lambda * dir_world;

            worldXY_all(p,:,partIdx) = world_point(1:2);  % X, Y
            partIdx = partIdx + 1;
        end
    end
end

