get_features folder contains data_process.m script to ingest pose data to extract the features. 
'features'; features{t} is an Nx4 matrix that stores all the detections at frame 't', columns are [ ID, x, y, orientation ]
orientation can be extracted from head, shoulders or hips, which can be specified in the data_process.m script. 

groundtruth folder contains the FF annotations for the segments per camera. One can merge the segments files. 

sample_data folder contains sample data extracted from pose data. features.mat, filtered_features.mat and groundtruth.mat correspond to the time segment from (vid2 seg8 - vid3seg 6, continuous segment) for camera 6 in this case. clean_gt.m removes persons in the scenes that are not in groundtruth (because the F-formations are annotated with the best camera view) from features.mat to obtain filtered_features.mat. Filtered_features.mat and groundtruth.mat are now ready as inputs to GCFF and GTCG. 