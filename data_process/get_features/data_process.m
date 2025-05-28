clear variables; close all

disp("**identify files**")

pose_data_path = ['/home/zonghuan/tudelft/projects/datasets/conflab/' ...
    'annotations/pose/coco/'];
Files=dir([pose_data_path, '*.json']); % edit your own path to the pose data!!!
addpath('../../utils/');
orient_choices = ["head", "shoulder", "hip", "foot"];

for orient_choice = orient_choices
save_path = "../../data/in_process/";
mkdir(sprintf(save_path + orient_choice));
mkdir(save_path + orient_choice + "/seg2/");
mkdir(save_path + orient_choice + "/seg3/");

imgSize = [1920, 1080];
num_kps = 20;
for k=1:length(Files)
    disp("***filenumber****")
    k
    FileName=Files(k).name;
    path = strcat(pose_data_path,FileName); % edit your own path to the pose data!!!
    
    % if k<21
    %     continue;
    % end
    data = jsondecode(fileread(path));
    annotations = data.annotations;
    disp("loaded annotations")

    if ~isstruct(data.annotations.skeletons)
        continue;
    end

    full_timestamps = uint64(1:1:length(data.annotations.skeletons));
    L = length(full_timestamps);
    timestamps = full_timestamps(1:59.96:end);

    colNames = fieldnames(data.annotations.skeletons);
    total_people_no = length(colNames);

    disp("enter time loop")
    features = cell(1, length(timestamps));
    for ti = 1:length(timestamps)
        t = timestamps(ti);
        frame_data = zeros(total_people_no, 4+num_kps);
        bp_data = zeros(total_people_no, 4+num_kps);
        for p = 1:total_people_no
            % Read keypoints
            headX = data.annotations.skeletons(t).(colNames{p}).keypoints(1);
            headY = data.annotations.skeletons(t).(colNames{p}).keypoints(2);
            noseX = data.annotations.skeletons(t).(colNames{p}).keypoints(3);
            noseY = data.annotations.skeletons(t).(colNames{p}).keypoints(4);

            leftShoulderX = data.annotations.skeletons(t).(colNames{p}).keypoints(13);
            leftShoulderY = data.annotations.skeletons(t).(colNames{p}).keypoints(14);
            rightShoulderX = data.annotations.skeletons(t).(colNames{p}).keypoints(7);
            rightShoulderY = data.annotations.skeletons(t).(colNames{p}).keypoints(8);

            leftHipX = data.annotations.skeletons(t).(colNames{p}).keypoints(25);
            leftHipY = data.annotations.skeletons(t).(colNames{p}).keypoints(26);
            rightHipX = data.annotations.skeletons(t).(colNames{p}).keypoints(19);
            rightHipY = data.annotations.skeletons(t).(colNames{p}).keypoints(20);

            leftFootX = data.annotations.skeletons(t).(colNames{p}).keypoints(33);
            leftFootY = data.annotations.skeletons(t).(colNames{p}).keypoints(34);
            rightFootX = data.annotations.skeletons(t).(colNames{p}).keypoints(31);
            rightFootY = data.annotations.skeletons(t).(colNames{p}).keypoints(32);

            leftAnkleX = data.annotations.skeletons(t).(colNames{p}).keypoints(29);
            leftAnkleY = data.annotations.skeletons(t).(colNames{p}).keypoints(30);
            rightAnkleX = data.annotations.skeletons(t).(colNames{p}).keypoints(23);
            rightAnkleY = data.annotations.skeletons(t).(colNames{p}).keypoints(24);

            % Store keypoints
            frame_data(p,5) = headX;
            frame_data(p,6) = headY;
            frame_data(p,7) = noseX;
            frame_data(p,8) = noseY;

            frame_data(p,9) = leftShoulderX;
            frame_data(p,10) = leftShoulderY;
            frame_data(p,11) = rightShoulderX;
            frame_data(p,12) = rightShoulderY;

            frame_data(p,13) = leftHipX;
            frame_data(p,14) = leftHipY;
            frame_data(p,15) = rightHipX;
            frame_data(p,16) = rightHipY;

            frame_data(p,17) = leftAnkleX;
            frame_data(p,18) = leftAnkleY;
            frame_data(p,19) = rightAnkleX;
            frame_data(p,20) = rightAnkleY;

            frame_data(p,21) = leftFootX;
            frame_data(p,22) = leftFootY;
            frame_data(p,23) = rightFootX;
            frame_data(p,24) = rightFootY;

            % Back Projection
            cam = str2double(FileName(4));
            cp = loadCamParams(cam);
            feat = backProject(frame_data, cp.K, cp.R, cp.t, cp.distCoeff, ...
                cp.bodyHeight, cp.img_size, cp.height_ratios_map, cp.part_column_map);
            bp_data(:, 5:end) = reshape(feat, [], num_kps);

            person_id = data.annotations.skeletons(t).(colNames{p}).id;
            frame_data = process_kp(frame_data, p, person_id, orient_choice, imgSize, false);
            bp_data = process_kp(bp_data, p, person_id, orient_choice, [1,1], true);

            % frame_data.size = # of people * 4
            % head orientation is not recorded. Instead, use headX and
            % headY.
            c = 9;

        end
        features{1,ti} = [frame_data, bp_data];

    end
    % subsample
    % timestamps = timestamps(1:59.96:end);
    % features = features(1:59.96:end);

    % saving
    fn = FileName(1:end-10);
    mat_name = fn + "_" + orient_choice + ".mat";
    % ts_name = fn + "_" + orient_choice + ".mat";
    % mkdir(sprintf(fn))
    segn = "seg" + get_seg_num(fn);
    save_name = orient_choice + "/" + segn + "/" + mat_name;
    assert(length(features) == length(timestamps));
    % save(save_path + save_name, 'features', 'timestamps')
end
end

%% Extract frames
% seg_file = 2;
% ts_path = "C:\Users\zongh\OneDrive - Delft University of Technology\" + ...
%     "tudelft\datasets\conflab\shoulder\seg" + seg_file + "\";
% Files=dir(ts_path + "*.mat"); % edit your own path to the pose data!!!
% allFrames = cell(0, 3);
% for k=1:length(Files)
%     data = load(ts_path + Files(k).name);
%     video = extractFramesFromVideo(Files(k).name, data.timestamps);
%     allFrames = [allFrames; video];
%     c = 9;
% end
% save("frames_seg" +seg_file + ".mat", "allFrames", '-v7.3');

%%

function kps = process_kp(kps, p, person_id, orient_choice, imgSize, rh_axis)
headX = kps(p,5);
headY = kps(p,6);
noseX = kps(p,7);
noseY = kps(p,8);

leftShoulderX = kps(p,9);
leftShoulderY = kps(p,10);
rightShoulderX = kps(p,11);
rightShoulderY = kps(p,12);

leftHipX = kps(p,13);
leftHipY = kps(p,14);
rightHipX = kps(p,15);
rightHipY = kps(p,16);

leftAnkleX = kps(p,17);
leftAnkleY = kps(p,18);
rightAnkleX = kps(p,19);
rightAnkleT = kps(p,20);

leftFootX = kps(p,21);
leftFootY = kps(p,22);
rightFootX = kps(p,23);
rightFootY = kps(p,24);

% Process from keypoints
head_vector = [(noseX-headX),(noseY-headY)].* imgSize;
shoulder_vector = [(leftShoulderX-rightShoulderX),(leftShoulderY-rightShoulderY)].* imgSize;
hip_vector = [(leftHipX-rightHipX),(leftHipY-rightHipY)].* imgSize;
foot_vector = [(leftFootX-rightFootX),(leftFootY-rightFootY)].* imgSize;

if rh_axis
    axis_mod = -1;
else
    axis_mod = 1;
end

shoulder_orient = [-shoulder_vector(:,2),(shoulder_vector(:,1))] * axis_mod;
hip_orient = [-hip_vector(:,2),(hip_vector(:,1))] * axis_mod;
foot_orient = [-foot_vector(:,2),(foot_vector(:,1))] * axis_mod;

vector_check = [head_vector; shoulder_orient; hip_orient; foot_orient];
vector_check = reverse_incorrect_vectors(vector_check);

head_vector = vector_check(1, :);
shoulder_orient = vector_check(2, :);
hip_orient = vector_check(3, :);
foot_orient = vector_check(4, :);
% Head vector is special. Head -> Nose is the same direction as
% body orientation
if orient_choice == "head"
    body_vector = head_vector;
    body_pos = [headX, headY];
    % Otherwise, body vector is perpendicular to the R->L (counterclockwise 90 degrees)
    % MODIFIED: If R->L is (x,y), then orientation should be
    % (-y,x)! The y-axis in images is from top to bottom, which is
    % different from usual right-hand coordinate system.
elseif orient_choice == "shoulder"
    body_vector = shoulder_orient;
    body_pos = [leftShoulderX + rightShoulderX, leftShoulderY + rightShoulderY] / 2;
elseif orient_choice == "hip"
    body_vector = hip_orient;
    body_pos = [leftHipX + rightHipX, leftHipY + rightHipY] / 2;
elseif orient_choice == "foot"
    body_vector = foot_orient;
    body_pos = [leftFootX + rightFootX, leftFootY + rightFootY] / 2;
end

% dotProduct = dot(head_vector(:),body_vector(:));
% if (dotProduct<0)
%     body_vector(:) = [-body_vector(1),-body_vector(2)];
% elseif(dotProduct==0)
%     disp('warning: head and body exactly perpendicular')
% end

head_orientation = FixRangeOfAngles(get_angle(head_vector));
body_orientation = FixRangeOfAngles(get_angle(body_vector));

% person id, position X, position Y, orientation
kps(p,1) = person_id;
kps(p,2) = body_pos(1)*imgSize(1);
kps(p,3) = body_pos(2)*imgSize(2);
kps(p,4) = body_orientation;
end

function [x, y, z] = extractVideoNum(nameString)
    tokens = regexp(nameString, 'cam(\d+)_vid(\d+)_seg(\d+)', 'tokens');
    if isempty(tokens)
        error('Invalid format for nameString. Expected format: camx_vidy_segz');
    end
    x = tokens{1}{1};
    y = tokens{1}{2};
    z = tokens{1}{3};
end

function frames = extractFramesFromVideo(nameString, timestamps)
% EXTRACTFRAMESFROMVIDEO Finds a video file and extracts specified frames.
%
% Inputs:
%   - nameString: A string of format "camx_vidy_segz".
%   - timestamps: A vector of frame numbers to extract.
%
% Output:
%   - frames: A concatenated cell array of extracted frames.
%
% Example Usage:
%   frames = extractFramesFromVideo("cam1_vid2_seg3", [10, 20, 30]);

    % Parse nameString to extract camX, vidY, segZ
    [x, y, z] = extractVideoNum(nameString);

    % Construct the expected video file path
    
    videoPathPattern = sprintf(['C:\\Users\\zongh\\OneDrive - Delft University of Technology\\' ...
        'tudelft\\datasets\\conflab\\video_segments\\cam%s\\vid%s-seg%s-*.mp4'], x, y, z);

    % Find the matching video file
    videoFiles = dir(videoPathPattern);
    if isempty(videoFiles)
        error('No matching video file found for %s', videoPathPattern);
    end

    % Take the first match (assuming only one correct file exists)
    videoFilePath = fullfile(videoFiles(1).folder, videoFiles(1).name);

    % Open the video file
    videoObj = VideoReader(videoFilePath);

    % Initialize frames cell array
    frames = cell(length(timestamps), 3);

    % Read frames at specified timestamps
    for i = 1:length(timestamps)
        frameNum = timestamps(i);

        % Ensure the frame number is within valid range
        if frameNum > 0 && frameNum <= videoObj.NumFrames
            videoObj.CurrentTime = double(frameNum - 1) / videoObj.FrameRate;
            frames{i, 1} = readFrame(videoObj);
        else
            warning('Frame number %d is out of range for video %s', frameNum, videoFilePath);
            frames{i, 1} = [];
        end
        frames{i, 2} = nameString(1:14);
        frames{i, 3} = frameNum;
    end
end


function n = get_seg_num(file_name)
    x = split(file_name,"_");
    x = x(2);
    n = x{1}(end);
end


