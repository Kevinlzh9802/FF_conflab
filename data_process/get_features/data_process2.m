clear variables; close all

disp("**identify files**")

pose_data_path = ['/home/zonghuan/tudelft/projects/datasets/conflab/' ...
    'annotations/pose/coco/'];
Files=dir([pose_data_path, '*.json']); % edit your own path to the pose data!!!

orient_choice = "head";
save_path = "../../data/in_process/";
mkdir(sprintf(save_path + orient_choice));
mkdir(save_path + orient_choice + "/seg2/");
mkdir(save_path + orient_choice + "/seg3/");

imgSize = [1920, 1080];
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

    timestamps = uint64(1:1:length(data.annotations.skeletons));
    L  = length(timestamps);
    colNames = fieldnames(data.annotations.skeletons);
    total_people_no = length(colNames);

    disp("enter time loop")
    for t = 1:length(timestamps)
        frame_data = zeros(total_people_no,4);
        for p = 1:total_people_no
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

            head_vector = [(noseX-headX),(noseY-headY)].* imgSize;
            shoulder_vector = [(leftShoulderX-rightShoulderX),(leftShoulderY-rightShoulderY)].* imgSize;
            hip_vector = [(leftHipX-rightHipX),(leftHipY-rightHipY)].* imgSize;
            foot_vector = [(leftFootX-rightFootX),(leftFootY-rightFootY)].* imgSize;
            % Head vector is special. Head -> Nose is the same direction as
            % body orientation
            if orient_choice == "head"
                body_vector = head_vector;
            % Otherwise, body vector is perpendicular to the R->L (counterclockwise 90 degrees)
            % MODIFIED: If R->L is (x,y), then orientation should be
            % (-y,x)! The y-axis in images is from top to bottom, which is
            % different from usual right-hand coordinate system.
            elseif orient_choice == "shoulder"
                body_vector = [-shoulder_vector(:,2),(shoulder_vector(:,1))]; 
            elseif orient_choice == "hip"
                body_vector = [-hip_vector(:,2),(hip_vector(:,1))];
            elseif orient_choice == "foot"
                body_vector = [-foot_vector(:,2),(foot_vector(:,1))];
            end

            dotProduct = dot(head_vector(:),body_vector(:));
            if (dotProduct<0)
                body_vector(:) = [-body_vector(1),-body_vector(2)];
            elseif(dotProduct==0)
                disp('warning: head and body exactly perpendicular')
            end

            head_orientation = FixRangeOfAngles(get_angle(head_vector));
            body_orientation = FixRangeOfAngles(get_angle(body_vector));

            % person id, position X, position Y, orientation
            frame_data(p,1) = data.annotations.skeletons(t).(colNames{p}).id;
            frame_data(p,2) = headX*1920;
            frame_data(p,3) = headY*1080;
            frame_data(p,4) = body_orientation;
            % frame_data.size = # of people * 4
            % head orientation is not recorded. Instead, use headX and
            % headY.

        end
    features{1,t} = frame_data;

    end
    % subsample
    timestamps = timestamps(1:59.96:end);
    features = features(1:59.96:end);

    % saving
    fn = FileName(1:end-10);
    mat_name = fn + "_" + orient_choice + ".mat";
    % ts_name = fn + "_" + orient_choice + ".mat";
    % mkdir(sprintf(fn))
    segn = "seg" + get_seg_num(fn);
    % save_name = orient_choice + "/" + segn + "/" + mat_name;
    % save(save_path + save_name, 'features', 'timestamps')
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