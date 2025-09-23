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
addpath(genpath('../GCFF')); % add GCFF path for data loading

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
% load('../data/frames.mat', 'frames');

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

params.used_parts = ["229", "428", "429", "828", "829", ...
    "232", "233", "234", "235", "236", ...
    "431", "433", "434", ...
    "631", "632", "633", "634", "635", "636", ...
    "831", "832", "833", "834", "835"];

% params.used_parts = ["629"];

% params.cams = [4];
% params.vids = [3];
% params.segs = [5,6];

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

%% GTCG Parameters
param = setParams();

%% Main Processing
results = struct;
clues = ["head", "shoulder", "hip", "foot"];

for clue = clues
    fprintf('Processing %s...\n', clue);
    
    % Load data for this clue
    file_name = "../data/" + clue + ".mat";
    load(file_name, 'all_data');
    
    % Filter data similar to GCFF
    used_data = filterAndConcatTable(data_results, params.used_parts);
    
    % Run GTCG detection
    results.(clue) = GTCG_main(param, clue, used_data, speaking_status);
    
    % Store groups in the same format as GCFF
    f_name = clue + "Res";
    used_data.(f_name) = results.(clue).groups;
    results.(clue).original_data = used_data;
    data_results(used_data.id, f_name) = used_data.(f_name);
end

%% Run the same analysis pipeline as GCFF
run constructFormations.m;
% run detectSubFloor.m;
% run spatialScores.m;
run detectGroupNumBreakpoints.m;

% run plotGroups.m;
% run plotGroupsInfo.m;

%% GTCG Main Function
function results = GTCG_main(param, clue, used_data, speaking_status)
% GTCG_MAIN - Main function for GTCG group detection
% Modified to follow GCFF pipeline structure

clear all_data;           

%% Data preparation
% Filter data for this specific clue
f_name = clue + "Feat";
features = (used_data.(f_name))';
GTgroups = (used_data.GT)';

plot_debug = false;

%% Computation
precisions = [];
recalls = [];
TPs = [];
FPs = [];
FNs = [];

detections = [];
s_speaker = [];
group_sizes = [];
groups = cell(length(features), 1); % Store groups for each frame

% if isempty(gcp('nocreate'))
%     parpool;
% end

for f = 1:numel(features)
    if ~isempty(features{f})
        last_f = f + param.numFrames - 1;
        feat = features(f:last_f);                   %copy the frames
        
        fprintf(['******* Frames ' num2str(f:last_f) ' *******\n']);
        if plot_debug
            fig = figure;
            plotFrustums(feat{1}, param.frustum, fig);
        end
        
        try
            [frame_groups, ~, ~] = detectGroups(feat, param);
        catch
            frame_groups = [];
            warning("empty group detected!");
        end
        
        % Add singletons for person IDs not in any detected groups
        if ~isempty(feat{1}) & ~isempty(frame_groups)
            all_person_ids = feat{1}(:,1);  % Get all person IDs from the first frame
            detected_ids = [];  % Collect all IDs that appear in detected groups
            
            % Extract all person IDs from detected groups
            for g = 1:length(frame_groups)
                if ~isempty(frame_groups{g})
                    detected_ids = [detected_ids, frame_groups{g}];
                end
            end
            
            % Find person IDs that are not in any detected group
            missing_ids = setdiff(all_person_ids, detected_ids);
            
            % Add missing IDs as singleton groups
            for i = 1:length(missing_ids)
                frame_groups{end+1} = [missing_ids(i)];  % Wrap in array to maintain consistent structure
            end
        end
        
        % Store groups for this frame
        groups{f} = frame_groups;

        if param.show.weights > 0
            %display the weights
            figure(param.show.weights);
            bar(weights);
            title('Weights used in Eq 8 of ACCV');
        end

        detections = [detections; {frame_groups}];

        if param.show.groups > 0
            fr = f + param.numFrames - 1; %frame used as reference for the evaluation
            showGroups(fr, frame_groups, GTgroups, param);
        end
        
        GT = GTgroups(:, f + param.numFrames - 1);
        [p, r, tp, fp, fn] = evalgroups(frame_groups, GT, param.evalMethod);

        %add the evaluation results to the queue
        precisions = [precisions; p'];
        recalls = [recalls; r'];
        TPs = [TPs; tp'];
        FPs = [FPs; fp'];
        FNs = [FNs; fn'];

        %% record results
        f_info = table2struct(used_data(f, {'Cam', 'Vid', 'Seg', 'Timestamp'}));
        [sp_ids, cf_ids] = readSpeakingStatus(speaking_status, f_info.Vid, ...
            f_info.Seg, 1, 1);
        [speaking, confidence] = readSpeakingStatus(speaking_status, f_info.Vid, ...
            f_info.Seg, f_info.Timestamp+1, 1);

        if speaking == -1000
            continue;
        end
        ss = getStatusForGroup(sp_ids, speaking, groups{f});
        % record group sizes and speaker info
        if ~isempty(groups{f})
            % ssg = zeros(length(groups), 1);
            % g_size = zeros(length(groups), 1);
            for k=1:length(ss)
                ssg = sum(ss{k});
                if ~(ssg > 10 || ssg < 0)
                    group_sizes = [group_sizes; length(groups{f}{k})];
                    s_speaker = [s_speaker; ssg];
                end
            end

        end
        result_name = clue + "Res";
        used_data.(result_name){f} = groups{f}';

        showResults(precisions, recalls);
    end
end

%% Store results
results = struct;
results.dataset = 'GTCG_runner2';
results.TP = TPs;
results.FPs = FPs;
results.detections = detections;
results.FNs = FNs;
results.precisions = precisions;
results.recalls = recalls;
results.body_orientations = clue;
results.group_sizes = group_sizes;
results.s_speaker = s_speaker;
results.original_data = used_data;
results.groups = groups; % Store groups for each frame

% saving_name = strcat('results_', dataset);
% save(saving_name,'results');

end
