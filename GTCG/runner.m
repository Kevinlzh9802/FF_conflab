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

%% ALGORITHM PARAMETERS


param.evalMethod='card';                    %'card' require that 2/3 of individals are correctly matched in a group
                                            %'all'  require that 3/3 of individals are correctly matched in a group (more stricter evaluation)

%multi/single frame
param.numFrames=1;                          %number of frames to analyze (>1 imply multiframe analysis)

%parameter for the 2D histogram
param.hist.n_x=20;                          %number of rows for the frustum descriptor
param.hist.n_y=20;                          %number of columns for the frustum descriptor

%displaying options
param.show.weights=0;                       %show the weight used to condense the similarity matrices
param.show.groups=0;                        %show a figure with the current frame, the decetion and the groundtruth
param.show.frustum=1;                       %show the frustum
param.show.results=1;                       %display the precision/recall/F1-score values

%weight calculation parameters
param.weight.mode='MOLP';                   %the multiframe mode is activated only if param.numFrames>1. Set to:
                                            %'MOLP' (MultiObjectiveLinearProgramming)
                                            %'EQUAL' use equal weights for the frames
                                            %'MAXENTROPY' pick the
                                            %combination that maximize the entropy of the weight

results = struct;
for clue = ["foot", "hip", "shoulder", "head"]
    results.(clue) = GTCG_main(param, clue);
end

a = figure;
for clue = ["foot", "hip", "shoulder", "head"]
    plotUniqueVals(results.(clue).group_sizes, a, true, clue);
    hold on;
end
hold off
legend

b = figure;
for k=4:7
    subplot(2,2,k-3);
    for clue = ["foot", "hip", "shoulder", "head"]
        gs = results.(clue).group_sizes;
        sp = results.(clue).s_speaker;
        card = gs(gs == k);
        card_ss = sp(gs == k);
        card_ss = card_ss(card_ss <= k);
        card_ss = card_ss(card_ss >= 0);
        % plotUniqueVals(results.(clue).s_speaker, ab);

        [C,~,ic] = unique(card_ss);
        a_counts = accumarray(ic,1);
        normalized = true;
        % figure(fig);
        if normalized
            count_normalized = a_counts / length(card_ss);
            plot(C, count_normalized, 'DisplayName',clue);
        else
            plot(C, a_counts);
        end
        hold on;
    end
    legend
end
hold off
% legend
c = 0;

function results = GTCG_main(param, clue)
clear all_data;
% dataset directory
dataset = 'sample_data';
datasetDir=strcat('data/',dataset); % edit your own path!
run([datasetDir '/dsetParameter.m']); %load the dataset parameters

%set the frustum modality
frustumMode='CVIU';                         %'CVIU' use the CVIU model (cite [1])
                                            %'ACCV' use the ACCV model (cite [2])

seqDir=''; %if a sub-sequence exists write the folder name here
datasetDir=[datasetDir seqDir];

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
% load('../data/filtered/frames.mat', 'frames');

used_data = filterTable(all_data, [6,8], [2,3], 'all');
GTgroups = (used_data.GT)';
features = (used_data.Features)';

%% Computation
precisions=[];
recalls=[];
TPs=[];
FPs=[];
FNs=[];

detections=[];
s_speaker = [];
group_sizes = [];

for f=1:numel(features)
    if ~isempty(features{f})
        last_f = f+param.numFrames-1;
        feat=features(f:last_f);                   %copy the frames
        
        fprintf(['******* Frames ' num2str(f:last_f) ' *******\n']);

        [groups, frustums,weights]=detectGroups(feat,param);    %detect groups
        info = table2struct(used_data(last_f, 2:5));
        % kp = 5;
        % if floor (f / (numel(features)/kp)) ~= floor ((f+1)/ (numel(features)/kp))
        %     % fig = figure;
        %     % plotFrustums(feat{1}, param.frustum, fig);
        %     % img = findMatchingFrame(used_data, frames, last_f);
        %     img= 0;
        %     
        %     [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, info.Vid, info.Seg, 1);
        %     [speaking, confidence] = readSpeakingStatus(speaking_status, info.Vid, info.Seg, info.Timestamp);
        % 
        %     disp_info = struct();
        %     disp_info.GT = GTgroups{last_f};
        %     disp_info.speaking = getStatusForGroup(sp_ids, speaking, GTgroups{last_f});
        %     disp_info.confidence = getStatusForGroup(cf_ids, confidence, GTgroups{last_f});
        %     disp_info.kp = readPoseInfo(info, feat{1}(:,1));
        % 
        %     plotFrustumsWithImage(feat{1}, param.frustum, img, disp_info);
        %     % disp(GTgroups{f});
        % end

        % if param.show.weights>0
        %     %display the weights
        %     figure(param.show.weights);
        %     bar(weights);
        %     title('Weights used in Eq 8 of ACCV');
        % end

        detections=[detections ; {groups}];

        if param.show.groups>0
            fr=f+param.numFrames-1; %frame used as reference for the evaluation
            showGroups(fr,groups,GTgroups,param);
        end

        [p,r,tp,fp,fn] = evalgroups(groups,GTgroups(:,f+param.numFrames-1),param.evalMethod);

        %add the evaluation results to the queue
        precisions=[precisions ; p'];
        recalls=[recalls ; r'];
        TPs=[TPs ; tp'];
        FPs=[FPs ; fp'];
        FNs=[FNs ; fn'];

        [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, info.Vid, info.Seg, 1);
        [speaking, confidence] = readSpeakingStatus(speaking_status, info.Vid, info.Seg, info.Timestamp);
        ss = getStatusForGroup(sp_ids, speaking, groups);
        % g_size = [g_size, length(groups)];
        if ~isempty(groups)
            ssg = zeros(length(groups), 1);
            g_size = zeros(length(groups), 1);
            for k=1:length(ss)
                g_size(k) = length(groups{k});
                ssg(k) = sum(ss{k});
            end
            group_sizes = [group_sizes; g_size];
            s_speaker = [s_speaker; ssg];
        end
        showResults(precisions,recalls);
    
    end
end

% disp(sum(s_speaker) / length(s_speaker));

% a = figure;
% plotUniqueVals(group_sizes, a);

% for k=4:7
%     card = group_sizes(group_sizes == k);
%     card_ss = s_speaker(group_sizes == k);
%     [C,ia,ic] = unique(group_sizes);
%     a_counts = accumarray(ic,1);
%     plot(C, a_counts);
% end

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

% saving_name = strcat('results_',dataset);
% save(saving_name,'results');

end
