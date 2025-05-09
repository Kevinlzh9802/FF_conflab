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
clear variables, close all;
addpath(genpath('../GCFF')) % add your own path

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
seqpath = 'data/sample_data';
load(fullfile(seqpath, "filtered_features.mat"));
load(fullfile(seqpath, "groundtruth.mat"));
load(fullfile(seqpath, "settings_gc.mat"));
addpath("../utils/");
param.frustum.length = 275;
param.frustum.aperture = 160;

%% Computing

% If only some frames are annotated, delete all the others from features.
[~,indFeat] = intersect(timestamp,GTtimestamp) ;
timestamp = timestamp(indFeat) ;
features  = features(indFeat) ;

% Initialize evaluation variables
TP = zeros(1,length(timestamp)) ;
FP = zeros(1,length(timestamp)) ;
FN = zeros(1,length(timestamp)) ;
precision = zeros(1,length(timestamp)) ;
recall = zeros(1,length(timestamp)) ;


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

end

pr = mean(precision) ;
re = mean(recall) ;

F1 = 2 * pr .* re ./ (pr+re) ;


[~,indFeat] = intersect(timestamp,GTtimestamp) ;
pr_avg = mean(precision(indFeat)) ;
re_avg = mean(recall(indFeat)) ;
F1_avg = 2 * pr_avg * re_avg / ( pr_avg + re_avg ) ;
fprintf('Average Precision: -- %d\n',pr_avg)
fprintf('Average Recall: -- %d\n',re_avg)
fprintf('Average F1 score: -- %d\n',F1_avg)

results = struct;
results.dataset = seqpath;
results.precisions = precision(indFeat);
results.recalls = recall(indFeat);
results.F1s = F1;
results.F1_avg = F1_avg;
results.body_orientations = 'head';

saving_name = "results/";
% save(saving_name,'results');
