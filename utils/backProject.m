% Function to backproject pixel points onto a horizontal plane at specified height
function worldXY_all = backProject(frame_data, K, R, t, distCoeffs, ...
    bodyHeight, img_size, height_ratios_map, part_column_map)
% frame_data: Nx20 matrix where each row is a person with keypoint pixel data
% height_ratios_map: struct with field names corresponding to body parts, each with height ratio (0=head, 1=foot)
% bodyHeight: assumed person height in world coordinates
% K, R, t: camera intrinsic and extrinsic parameters
% distCoeffs: distortion coefficients [k1, k2, p1, p2, k3]

% bodyHeight = params.bodyHeight;
% img_size = params.imgSize;
% height_ratios_map = params.height_ratios_map;
% part_column_map = params.part_column_map;

numPeople = size(frame_data, 1);
body_parts = fieldnames(height_ratios_map);
numParts = numel(body_parts);

worldXY_all = zeros(numPeople, 2, numParts); % [X, Y] for each person and part

% Kinv = inv(K);
Rinv = R';
camCenter = -Rinv * t;

for p = 1:numPeople
    partIdx = 1;
    for part = body_parts'
        name = part{1};
        cols = part_column_map.(name);
        uv = (frame_data(p, cols) .* img_size)';  % [u; v]

        % Normalize using intrinsic matrix
        x = (uv(1) - K(1,3)) / K(1,1);
        y = (uv(2) - K(2,3)) / K(2,2);

        % Apply distortion correction
        r2 = x^2 + y^2;
        x_dist = x * (1 + distCoeffs(1)*r2 + distCoeffs(2)*r2^2 + distCoeffs(5)*r2^3) + 2*distCoeffs(3)*x*y + distCoeffs(4)*(r2 + 2*x^2);
        y_dist = y * (1 + distCoeffs(1)*r2 + distCoeffs(2)*r2^2 + distCoeffs(5)*r2^3) + distCoeffs(3)*(r2 + 2*y^2) + 2*distCoeffs(4)*x*y;

        dir_cam = [x_dist; y_dist; 1];
        dir_world = Rinv * dir_cam;

        Z = height_ratios_map.(name) * bodyHeight;
        lambda = (Z - camCenter(3)) / dir_world(3);
        world_point = camCenter + lambda * dir_world;

        % worldXY_all(p,:,partIdx) = world_point(1:2);  % X, Y
        worldXY_all(p,:,partIdx) = world_point(1:2);  % X, Y
        partIdx = partIdx + 1;
    end
end
end