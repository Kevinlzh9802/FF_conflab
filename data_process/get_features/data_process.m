clear
clc
close all

disp("**identify files**")
Files=dir('../processed/annotation/pose_v2/*.json'); % edit your own path to the pose data!!!

for k=1:length(Files)
    disp("***filenumber****")
    k
    FileName=Files(k).name;
    path = strcat('../annotation/pose_v2/',FileName); % edit your own path to the pose data!!!

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

            head_vector = [(noseX-headX),(noseY-headY)];
            shoulder_vector = [(leftShoulderX-rightShoulderX),(leftShoulderY-rightShoulderY)];
            hip_vector = [(leftHipX-rightHipX),(leftHipY-rightHipY)];
            foot_vector = [(leftFootX-rightFootX),(leftFootY-rightFootY)];
            body_vector = [shoulder_vector(:,2),-(shoulder_vector(:,1))]; % edit to your choice of head, shoulder, hip, foot!!! 
            
            
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
            frame_data(p,2) = headX*1960;
            frame_data(p,3) = headY*1080;
            frame_data(p,4) = body_orientation;


        end
    features{1,t} = frame_data;
    
    end
    % subsample
    timestamps = timestamps(1:59.96:end);
    features = features(1:59.96:end);
    
    saving
    folderName = FileName(1:end-10);
    mkdir(sprintf(folderName))
    
    save(strcat(sprintf(folderName),'/features.mat'),'features')
    save(strcat(sprintf(folderName),'/timestamps.mat'),'timestamps')
end




