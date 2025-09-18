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

%% original loading
% seqpath = 'data/sample_data';
% load(fullfile(seqpath, "filtered_features.mat"));
% load(fullfile(seqpath, "groundtruth.mat"));
% load(fullfile(seqpath, "settings_gc.mat"));

%% zonghuan loading
% load('../data/data_results.mat');
load('../data/speaking_status.mat', 'speaking_status');
load('../data/frames.mat', 'frames');

%% params
params.frustum.length = 275;
params.frustum.aperture = 160;
use_real = true;
if use_real
    params.stride = 70;
    params.mdl = 90000;
else
    params.stride = 130;
    params.mdl = 60000;
end

% params.used_parts = ["229", "428", "429", "828", "829", ...
%     "232", "233", "234", "235", "236", ...
%     "431", "433", "434", ...
%     "631", "632", "633", "634", "635", "636", ...
%     "831", "832", "833", "834", "835"];

params.used_parts = ["834"];

file_name = "../data/head.mat";
load(file_name, 'all_data');
data_results = all_data;
data_results.Properties.VariableNames{1} = 'headFeat';

for clue = ["head", "shoulder", "hip", "foot"]
    f_name = clue + "Feat";
    file_name = "../data/" + clue + ".mat";
    load(file_name, 'all_data');
    data_results.(f_name) = all_data.Features;
end
data_results = data_results(:, [1 7 8 9 2 3 4 5 6]);

% In image pixels, use_real is false, and isLeftHanded is true.
% Vice versa.
data_results = processFootData(data_results, ~use_real);
run concatSegs.m;
data_results.id = (1:height(data_results))';

results = struct;
clues = ["head", "shoulder", "hip", "foot"];
for clue = clues
    used_data = filterAndConcatTable(data_results, params.used_parts);    
    results.(clue) = GCFF_main(used_data, params, clue, speaking_status);

    f_name = clue + "Res";
    used_data.(f_name) = results.(clue).groups;
    results.(clue).original_data = used_data;
    data_results(used_data.id, f_name) = used_data.(f_name);
    % data_results.(f_name) = used_data.(f_name);
    % results.(clue).g_count = countGroupsContainingIDs(used_data.(f_name), {[13,21],[35,12,19]});
end

% analysis = data_results(~cellfun(@isempty, data_results.headRes) & data_results.concat_id, :);

% unique_segs = unique(data_results.Vid);
% formations = table;
% for u_ind=1:length(unique_segs)
%     u = unique_segs(u_ind);
%     ana = data_results(data_results.Vid == u, :);
%     unique_groups = recordUniqueGroups(ana, "headRes");
%     unique_groups.Vid = zeros(height(unique_groups), 1) + u;
%     formations = [formations; unique_groups];
% end

run constructFormations.m;
run detectSubFloor.m;
% run spatialScores.m;

run plotGroups.m;
run plotGroupsInfo.m;

%% Computing
function [results, data] = GCFF_main(data, params, clue, speaking_status)
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
use_real = false;

% Initialize evaluation variables
TP = zeros(1,length(timestamp));
FP = zeros(1,length(timestamp));
FN = zeros(1,length(timestamp));
precision = zeros(1,length(timestamp)) ;
recall = zeros(1,length(timestamp)) ;
s_speaker = [];
group_sizes = [];

for idxFrame = 1:length(timestamp)
    % gg represents group_id
    feat = features{idxFrame}(:, [1:24] + 24 * use_real);
    gg = gc(feat(:, 1:4), stride, mdl);

    groups{idxFrame} = [] ;
    for ii = 1:max(gg)+1
        groups{idxFrame}{ii} = feat(gg==ii-1,1) ;
    end

    if ~isempty(groups{idxFrame})
        groups_temp = ff_deletesingletons(groups{idxFrame});
        if isempty(groups_temp)
            groups{idxFrame} = [];
        end
    end
    if ~isempty(GTgroups{idxFrame})
        GTgroups{idxFrame} = ff_deletesingletons(GTgroups{idxFrame}) ;
    end
    [precision(idxFrame),recall(idxFrame),TP(idxFrame),FP(idxFrame),FN(idxFrame)] = ff_evalgroups(groups{idxFrame},GTgroups{idxFrame},'card') ;
    

    % DISPLAY RESULTS
    displayFrameResults(idxFrame, length(timestamp), groups{idxFrame}, GTgroups{idxFrame});

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
        disp_info.kp = readPoseInfo(f_info, feat(:,1));

        plotFrustumsWithImage(feat, params.frustum, img, disp_info);
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

function displayFrameResults(idxFrame, totalFrames, groups, GTgroups)
% DISPLAY RESULTS
% Display frame information, found groups, and ground truth groups
%
% Inputs:
%   idxFrame - current frame index
%   totalFrames - total number of frames
%   groups - detected groups for current frame
%   GTgroups - ground truth groups for current frame

% Frame:
fprintf('Frame: %d/%d\n', idxFrame, totalFrames)
% Found:
fprintf('   FOUND:-- ')
if ~isempty(groups)
    for ii=1:size(groups,2)
        fprintf(' %i',groups{ii})
        fprintf(' |')
    end
else
    fprintf(' No Groups ')
end
fprintf('\n')
% GT:
fprintf('   GT   :-- ')
if ~isempty(GTgroups)
    for ii=1: size(GTgroups,2)
        fprintf(' %i',GTgroups{ii})
        fprintf(' |')
    end
else
    fprintf(' No Groups ')
end
fprintf('\n');
end
