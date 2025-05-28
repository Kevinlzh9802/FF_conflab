
% Example visualization script
% Define camera parameters K, R, t beforehand
% Assume frame_data and height_ratios_map are available
% Corresponding vertical ratio (0 for head, 1 for feet)

%% Load camera params

function camParams = loadCamParams(cam)
camParams = struct;
intrinsic_path = "../../data/camera_params/intrinsic_" + cam + ".json";
extrinsic_path = "../../data/camera_params/extrinsic_zh_" + cam + ".json";
intrinsics = jsondecode(fileread(intrinsic_path));
extrinsics = jsondecode(fileread(extrinsic_path));

camParams.K = intrinsics.intrinsic;
camParams.distCoeff = intrinsics.distortion_coefficients;
camParams.R = extrinsics.rotation;
camParams.t = extrinsics.translation;

camParams.height_ratios_map = struct('head', 1, 'nose', 0.95, ...
    'leftShoulder', 0.85, 'rightShoulder', 0.85, ...
    'leftHip', 0.5, 'rightHip', 0.5, ...
    'leftAnkle', 0.02, 'rightAnkle', 0.02, ...
    'leftFoot', 0.02, 'rightFoot', 0.02);

% Mapping from frame_data columns to part names
camParams.part_column_map = struct('head', [5,6], 'nose', [7,8], ...
    'leftShoulder', [9,10], 'rightShoulder', [11,12], ...
    'leftHip', [13,14], 'rightHip', [15,16], ...
    'leftAnkle', [17,18], 'rightAnkle', [19,20], ...
    'leftFoot', [21,22], 'rightFoot', [23,24]);

camParams.bodyHeight = 170;
camParams.img_size = [1920, 1080];
end

%% Test
% load("../data/head.mat");
% used_data = filterTable(all_data, cam, 'all', 'all');
% frame_data = used_data.Features{142};
% worldXY_all = backProject(frame_data, K, R, t, distCoeff, params);

%% Visualization
% figure;
% hold on;
% numPeople = size(worldXY_all, 1);
% numParts = size(worldXY_all, 3);
% 
% colors = lines(numPeople);
% labels = fieldnames(height_ratios_map);
% 
% for i = 1:numPeople
%     for j = 1:numParts
%         plot(worldXY_all(i,1,j), worldXY_all(i,2,j), 'o', 'Color', colors(i,:), 'MarkerSize', 8);
%     end
% end
% 
% legend(labels, 'Interpreter', 'none');
% xlabel('World X');
% ylabel('World Y');
% title('Backprojected Keypoints on Ground Plane');
% grid on;
% axis equal;
