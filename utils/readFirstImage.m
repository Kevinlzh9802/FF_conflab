clear variables; close all;

video_path = "/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/cameras/video/cam02/GH010003.MP4";
vid = VideoReader(video_path);

% Read the first frame
firstFrame = readFrame(vid);

% Display the frame
imshow(firstFrame);
title('First Frame of the Video');