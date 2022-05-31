close all
clear all
clc

GT = load('groundtruth.mat');
features = load('features.mat');
featureGroups = features.features;
timestamp = GT.GTtimestamp;
GTgroups= GT.GTgroups;

% eliminate people that don't appear in GT
for i = 1:length(timestamp)
    gt_group = GTgroups{1,i};
    group_participants = horzcat(gt_group{:});
    
    feature_group = featureGroups{1,i};
    video_participants = feature_group(:,1);
    
    [sharedvals,idx] = intersect(video_participants,group_participants,'stable');
    filtered_feature_group = feature_group(idx,:);
    
    c = setdiff(filtered_feature_group(:,1),group_participants);
    
    if ~isempty(c)
        adjustment = zeros(length(c),4);
        for m = 1:length(c)
            adjustment(m,1) = c(m);
        end
        filtered_feature_group = [filtered_feature_group;adjustment];
    end
    filtered_feature_group(isnan(filtered_feature_group))=0;
    featureGroups{1,i} = filtered_feature_group;

end
FoV = features.FoV;
timestamp = features.timestamp;
features = featureGroups;


