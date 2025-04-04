function kp_data = readPoseInfo(info, person)

pose_path = "/home/zonghuan/tudelft/projects/datasets/conflab/annotations/pose/coco/";
file_name =  "cam"+info.Cam + "_vid"+info.Vid + "_seg"+info.Seg + "_coco.json";
data = jsondecode(fileread(pose_path + file_name));

kp = data.annotations.skeletons(info.Timestamp);
kp_data = struct();
kp_x = [1,3,13,7,25,19,33,31];
kp_y = kp_x + 1;
for k=1:length(person)
    col_name = "x" + person(k);
    kp_data.(col_name) = zeros(8, 2);
    kp_data.(col_name)(:,1) = kp.(col_name).keypoints(kp_x);
    kp_data.(col_name)(:,2) = kp.(col_name).keypoints(kp_y);
    % kp_data.headX = kp.(col_name).keypoints(1);
    % kp_data.headY = kp.(col_name).keypoints(2);
    % kp_data.noseX = kp.(col_name).keypoints(3);
    % kp_data.noseY = kp.(col_name).keypoints(4);
    % 
    % kp_data.leftShoulderX = kp.(col_name).keypoints(13);
    % kp_data.leftShoulderY = kp.(col_name).keypoints(14);
    % kp_data.rightShoulderX = kp.(col_name).keypoints(7);
    % kp_data.rightShoulderY = kp.(col_name).keypoints(8);
    % 
    % kp_data.leftHipX = kp.(col_name).keypoints(25);
    % kp_data.leftHipY = kp.(col_name).keypoints(26);
    % kp_data.rightHipX = kp.(col_name).keypoints(19);
    % kp_data.rightHipY = kp.(col_name).keypoints(20);
    % 
    % kp_data.leftFootX = kp.(col_name).keypoints(33);
    % kp_data.leftFootY = kp.(col_name).keypoints(34);
    % kp_data.rightFootX = kp.(col_name).keypoints(31);
    % kp_data.rightFootY = kp.(col_name).keypoints(32);
end

end