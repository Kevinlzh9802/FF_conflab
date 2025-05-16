% Sebastiano Vascon      Version 1.01
% Copyright 2016 Sebastiano Vascon.  [sebastiano.vascon-at-iit.it]
% Please email me if you have any questions.
%
% Please cite one of these works
% [1] S. Vascon, E. Zemene , M. Cristani, H. Hung, M.Pelillo and V. Murino
% Detecting conversational groups in images and sequences: A robust game-theoretic approach
% Computer Vision and Image Understanding (CVIU), 2016
%
% [2] S. Vascon, E. Zemene , M. Cristani, H. Hung, M.Pelillo and V. Murino
% A Game-Theoretic Probabilistic Approach for Detecting Conversational Groups
% ACCV 2014

% -------------------------------------------- %
%                Detect Groups in Single Frame %
% -------------------------------------------- %

% clear variables; 
clearvars -except frames;
close all;
warning off;

addpath(genpath('utils'));
addpath(genpath('libs'));
addpath(genpath('../utils'));
param = setParams();

results = struct;
for clue = ["hip", "head"]
    results.(clue) = GTCG_main(param, clue);
    f_name = clue + "Res";
    results.(clue).g_count = countGroupsContainingIDs(results.(clue).original_data.(f_name), {[13,21],[35,12,19]});
end
c = 9;
% run plotGroups.m;

function results = GTCG_main(param, clue)
clear all_data;           
% seqDir=''; %if a sub-sequence exists write the folder name here
% datasetDir=[datasetDir seqDir];

%% original loading
% load([datasetDir '/filtered_features.mat'],'features','timestamp');
% load([datasetDir '/groundtruth.mat'],'GTgroups','GTtimestamp');
% 
% % [~,indFeat] = intersect(timestamp,int64(GTtimestamp));
% [~,indFeat] = intersect(timestamp,GTtimestamp);
% timestamp = timestamp(indFeat);
% features  = features(indFeat);

%% Zonghuan loading
file_name = "../data/" + clue + ".mat";
load(file_name, 'all_data');
load('../data/speaking_status.mat', 'speaking_status');
% load('../data/frames.mat', 'frames');

used_data = filterTable(all_data, [4], [2], [8]);
GTgroups = (used_data.GT)';
features = (used_data.Features)';

plot_debug = false;
%% Computation
precisions=[];
recalls=[];
TPs=[];
FPs=[];
FNs=[];

detections=[];
s_speaker = [];
group_sizes = [];

% if isempty(gcp('nocreate'))
%     parpool;
% end

for f=18:numel(features)
    if ~isempty(features{f})
        last_f = f+param.numFrames-1;
        feat=features(f:last_f);                   %copy the frames
        
        fprintf(['******* Frames ' num2str(f:last_f) ' *******\n']);
        if plot_debug
            fig = figure;
            plotFrustums(feat{1}, param.frustum, fig);
        end
        [groups, ~, ~]=detectGroups(feat,param);

        % fDetect = parfeval(@detectGroups, 3, feat, param);
        % completed = wait(fDetect, "finished", 30);
        % if completed
        %     % Attempt to fetch results with a timeout
        %         %detect groups
        %     [groups, ~, ~] = fetchOutputs(fDetect);
        % else
        %     % Timeout or other error
        %     cancel(fDetect);
        %     warning('evalgroups at iteration %d timed out or failed. Skipping...', f);
        %     continue;
        % end

        info = table2struct(used_data(last_f, 2:5));
        kp = 5;
        plot_cond = (floor (f / (numel(features)/kp)) ~= floor ((f+1)/ (numel(features)/kp)));
        % plot_cond = true;
        if plot_cond
            % fig = figure;
            % plotFrustums(feat{1}, param.frustum, fig);
            % img = findMatchingFrame(used_data, frames, last_f);
            % % img= 0;
            % 
            % [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, info.Vid, info.Seg, 1, 1);
            % [speaking, confidence] = readSpeakingStatus(speaking_status, info.Vid, info.Seg, info.Timestamp, 1);
            % 
            % disp_info = struct();
            % disp_info.GT = GTgroups{last_f};
            % disp_info.detection = groups;
            % disp_info.speaking = getStatusForGroup(sp_ids, speaking, GTgroups{last_f});
            % disp_info.confidence = getStatusForGroup(cf_ids, confidence, GTgroups{last_f});
            % disp_info.kp = readPoseInfo(info, feat{1}(:,1));
            % 
            % plotFrustumsWithImage(feat{1}, param.frustum, img, disp_info);
            % disp(GTgroups{f});
        end

        if param.show.weights>0
            %display the weights
            figure(param.show.weights);
            bar(weights);
            title('Weights used in Eq 8 of ACCV');
        end

        detections=[detections ; {groups}];

        if param.show.groups>0
            fr=f+param.numFrames-1; %frame used as reference for the evaluation
            showGroups(fr,groups,GTgroups,param);
        end
        
        GT = GTgroups(:,f+param.numFrames-1);
        [p,r,tp,fp,fn] = evalgroups(groups, GT, param.evalMethod);

        %add the evaluation results to the queue
        precisions=[precisions ; p'];
        recalls=[recalls ; r'];
        TPs=[TPs ; tp'];
        FPs=[FPs ; fp'];
        FNs=[FNs ; fn'];

        [sp_ids, ~] = readSpeakingStatus(speaking_status, info.Vid, ...
            info.Seg, 1, 1);
        [speaking, ~] = readSpeakingStatus(speaking_status, info.Vid, ...
            info.Seg, info.Timestamp+1, 1);
        if speaking == -1000
            continue;
        end
        ss = getStatusForGroup(sp_ids, speaking, groups);
        
        % record group sizes and speaker info
        if ~isempty(groups)
            % ssg = zeros(length(groups), 1);
            % g_size = zeros(length(groups), 1);
            for k=1:length(ss)
                ssg = sum(ss{k});
                if ~(ssg > 10 || ssg < 0)
                    group_sizes = [group_sizes; length(groups{k})];
                    s_speaker = [s_speaker; ssg];
                end
            end
            
        end
        showResults(precisions,recalls);
        result_name = clue + "Res";
        used_data.(result_name){f} = groups;
    end
end

results = struct;
results.dataset = dataset;
results.TP = TPs;
results.FPs = FPs;
results.detections = detections;
results.FNs = FNs;
results.precisions = precisions;
results.recalls = recalls;
results.body_orientations = 'head';
results.group_sizes = group_sizes;
results.s_speaker = s_speaker;
results.original_data = used_data;

% saving_name = strcat('results_',dataset);
% save(saving_name,'results');

end
