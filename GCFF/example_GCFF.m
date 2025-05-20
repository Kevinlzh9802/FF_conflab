% -----------------------------------------
% Graph-Cuts for F-Formation (GCFF)
% 2015 - University of Verona
% Written by Francesco Setti
% -----------------------------------------
%
% This script is just an example of how to use the GCFF code to run
% experiments on provided data.
%

%% INITIALIZATION

% Clean the workspace
% clear variables, close all;
clearvars -except frames;
addpath(genpath('../GCFF')) % add your own path
addpath(genpath('../utils'));

%% Zonghuan exp
% % Set data folder
% seqpath = 'data/zh_exp';
% % NB: edit here your own path to data!!!
% cam = 6;
% seg = 3;
%
% % Load features
% featureFile = load(fullfile(seqpath, "seg" + seg + ".mat"));
% % Load groundtruth
% GTFile = load(fullfile(seqpath,"seg" + seg + "_gt.mat"));
% %Load settings
% load(fullfile(seqpath,'settings_gc.mat'));
%
% camField = sprintf('cam%d', cam);
% features = featureFile.features.(camField);
% GTgroups = GTFile.cameraData.(camField);
% timestamp = 1:length(features);
% GTtimestamp = 1:length(GTgroups);

%% original loading
% seqpath = 'data/sample_data';
% load(fullfile(seqpath, "filtered_features.mat"));
% load(fullfile(seqpath, "groundtruth.mat"));
% load(fullfile(seqpath, "settings_gc.mat"));

%% zonghuan loading
% load('../data/data_results.mat');
load('../data/speaking_status.mat', 'speaking_status');
% load('../data/frames.mat', 'frames');

%% params
params.frustum.length = 275;
params.frustum.aperture = 160;
params.stride = 130;
params.mdl = 60000;
params.cams = [6];
params.vids = [3];
params.segs = [5];

file_name = "../data/head.mat";
load(file_name, 'all_data');
data_results = all_data;
data_results.Properties.VariableNames{1} = 'headFeat';

for clue = ["shoulder", "hip", "foot"]
    f_name = clue + "Feat";
    file_name = "../data/" + clue + ".mat";
    load(file_name, 'all_data');
    data_results.(f_name) = all_data.Features;
end
data_results = data_results(:, [1 7 8 9 2 3 4 5 6]);

results = struct;
for clue = ["head", "shoulder", "hip", "foot"]
    used_data = filterTable(data_results, params.cams, params.vids, params.segs);
    results.(clue) = GCFF_main(used_data, params, clue, speaking_status, frames);

    f_name = clue + "Res";
    used_data.(f_name) = results.(clue).groups;
    results.(clue).original_data = used_data;
    % data_results.(f_name) = used_data.(f_name);
    % results.(clue).g_count = countGroupsContainingIDs(used_data.(f_name), {[13,21],[35,12,19]});
end

run plotGroups.m;
% run plotGroupsInfo.m;

%% Computing
function results = GCFF_main(data, params, clue, speaking_status, frames)
% If only some frames are annotated, delete all the others from features.
% [~,indFeat] = intersect(timestamp,GTtimestamp) ;
% timestamp = timestamp(indFeat) ;
% features  = features(indFeat) ;

f_name = clue + "Feat";
features = (data.(f_name))';
GTgroups = (data.GT)';
timestamp = data.Timestamp; 
stride = params.stride;
mdl = params.mdl;

% Initialize evaluation variables
TP = zeros(1,length(timestamp)) ;
FP = zeros(1,length(timestamp)) ;
FN = zeros(1,length(timestamp)) ;
precision = zeros(1,length(timestamp)) ;
recall = zeros(1,length(timestamp)) ;
s_speaker = [];
group_sizes = [];

for idxFrame = 1:length(timestamp)
    % gg represents group_id
    gg = gc( features{idxFrame}, stride, mdl ) ;
    groups{idxFrame} = [] ;
    for ii = 1:max(gg)+1
        groups{idxFrame}{ii} = features{idxFrame}(gg==ii-1,1) ;
    end

    if ~isempty(groups{idxFrame})
        groups{idxFrame} = ff_deletesingletons(groups{idxFrame}) ;
    end
    if ~isempty(GTgroups{idxFrame})
        GTgroups{idxFrame} = ff_deletesingletons(GTgroups{idxFrame}) ;
    end
    [precision(idxFrame),recall(idxFrame),TP(idxFrame),FP(idxFrame),FN(idxFrame)] = ff_evalgroups(groups{idxFrame},GTgroups{idxFrame},'card') ;
    

    % DISPLAY RESULTS
    % Frame:
    fprintf('Frame: %d/%d\n', idxFrame, length(timestamp))
    % Found:
    fprintf('   FOUND:-- ')
    if ~isempty(groups{idxFrame})
        for ii=1:size(groups{idxFrame},2)
            fprintf(' %i',groups{idxFrame}{ii})
            fprintf(' |')
        end
    else
        fprintf(' No Groups ')
    end
    fprintf('\n')
    % GT:
    fprintf('   GT   :-- ')
    if ~isempty(GTgroups{idxFrame})
        for ii=1: size(GTgroups{idxFrame},2)
            fprintf(' %i',GTgroups{idxFrame}{ii})
            fprintf(' |')
        end
    else
        fprintf(' No Groups ')
    end
    fprintf('\n');

    %% record results
    f_info = table2struct(data(idxFrame, {'Cam', 'Vid', 'Seg', 'Timestamp'}));
    [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, f_info.Vid, ...
        f_info.Seg, 1, 1);
    [speaking, confidence] = readSpeakingStatus(speaking_status, f_info.Vid, ...
        f_info.Seg, f_info.Timestamp+1, 1);
    
    if speaking == -1000
        continue;
    end
    ss = getStatusForGroup(sp_ids, speaking, groups{idxFrame});
    % record group sizes and speaker info
    if ~isempty(groups{idxFrame})
        % ssg = zeros(length(groups), 1);
        % g_size = zeros(length(groups), 1);
        for k=1:length(ss)
            ssg = sum(ss{k});
            if ~(ssg > 10 || ssg < 0)
                group_sizes = [group_sizes; length(groups{idxFrame}{k})];
                s_speaker = [s_speaker; ssg];
            end
        end

    end
    result_name = clue + "Res";
    data.(result_name){idxFrame} = groups{idxFrame};

    %% Plot
    plot_cond = false;
    if plot_cond
        img = findMatchingFrame(data, frames, idxFrame);
        
        disp_info = struct();
        disp_info.GT = GTgroups{idxFrame};
        disp_info.detection = groups{1};
        disp_info.speaking = getStatusForGroup(sp_ids, speaking, GTgroups{idxFrame});
        disp_info.confidence = getStatusForGroup(cf_ids, confidence, GTgroups{idxFrame});
        disp_info.kp = readPoseInfo(f_info, features{idxFrame}(:,1));

        plotFrustumsWithImage(features{idxFrame}, params.frustum, img, disp_info);
        disp(GTgroups{idxFrame});
    end
    

end

pr = mean(precision) ;
re = mean(recall) ;

F1 = 2 * pr .* re ./ (pr+re) ;


% [~,indFeat] = intersect(timestamp,GTtimestamp) ;
indFeat = 1:length(timestamp);
pr_avg = mean(precision(indFeat)) ;
re_avg = mean(recall(indFeat)) ;
F1_avg = 2 * pr_avg * re_avg / ( pr_avg + re_avg ) ;
fprintf('Average Precision: -- %d\n',pr_avg)
fprintf('Average Recall: -- %d\n',re_avg)
fprintf('Average F1 score: -- %d\n',F1_avg)

results = struct;
% results.dataset = seqpath;
results.precisions = precision(indFeat);
results.recalls = recall(indFeat);
results.F1s = F1;
results.F1_avg = F1_avg;
results.body_orientations = 'head';
results.groups = data.(result_name);
results.group_sizes = group_sizes;
results.s_speaker = s_speaker;

saving_name = "results/";
% save(saving_name,'results');
end
% 
% function [sps] = get_speaking_status(data)
% for idxFrame = 1:length(timestamp)
% end
% end