clear variables; close all;

feature_folder = "../data/in_process/";

for orient_choice = ["head", "shoulder", "hip", "foot"]
    all_data = {};
    for seg=2:3
        feature_path = feature_folder + orient_choice + "/seg" + seg + "/";
        gt_path = "./groundtruth/seg" + seg + "_gt.mat";
        
        GT = load(gt_path);
        % GT = load('groundtruth.mat');
        features = concatCamFeatures(feature_path);
        
        for c=2:2:8
            camField = sprintf('cam%d', c);
            % featureGroups = features.(camField);
% 
            % GTgroups = (GT.cameraData.(camField))';
            % [featureGroups, GTgroups] = get_intersects(featureGroups, GTgroups);
            % 
            % timestamp = 1:length(GTgroups);
            % features.(camField) = clear_people(featureGroups, GTgroups, timestamp);

            % features.(camField) = featureGroups;
            GTgroups = cell(height(features.(camField)), 1);
            all_data = [all_data; features.(camField), GTgroups];
        end
    end
    all_data = cell2table(all_data, 'VariableNames', ["Features", ...
        "Cam", "Vid", "Seg", "Timestamp", "GT"]);
    save("../data/filtered/" + orient_choice + ".mat", "all_data");
end

% featureGroups = features.features;
% timestamp = GT.GTtimestamp;
% GTgroups= GT.GTgroups;

function [features, GT] = get_intersects(features, GT)
    min_len = min(length(features), length(GT));
    features = features(1:min_len, :);
    GT = GT(1:min_len, :);
end

function featureGroups = clear_people(featureGroups, GTgroups, timestamp)
% eliminate people that don't appear in GT
for i = 1:length(timestamp)
    gt_group = GTgroups{i, 1};
    group_participants = horzcat(gt_group{:});
    
    feature_group = featureGroups{i, 1};
    video_participants = feature_group(:,1);
    
    [~,idx] = intersect(video_participants,group_participants,'stable');
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
    featureGroups{i, 1} = filtered_feature_group;

end
% FoV = features.FoV;
% timestamp = features.timestamp;
% features = featureGroups;
end


function camData = concatCamFeatures(folderPath)
% CONCATCAMFEATURES Read and concatenate features from .mat files of the form:
%   camX_vidY_segZ_*.mat,
% where X, Y, Z are integers. Each file contains:
%   features (cell array)
%   timestamps (1D vector)
%
% Returns a struct CAMDATA where each camera index X has a field 'features'
% that contains all the concatenated features in ascending order of Y, then Z.
%
% USAGE:
%   camData = concatCamFeatures('C:\path\to\folder');

    % Find all .mat files in the folder
    files = dir(fullfile(folderPath, '*.mat'));
    
    % A containers.Map will store data grouped by camera X.
    % Key:   double (X)
    % Value: cell array of rows {Y, Z, featuresCell}
    dataMap = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    % Regular expression for filenames: camX_vidY_segZ_*.mat
    % Example: cam1_vid3_seg2_something.mat
    pattern = '^cam(\d+)_vid(\d+)_seg(\d+).*\.mat$';
    
    % Loop over each .mat file in the directory
    for f = 1:numel(files)
        fname = files(f).name;
        fpath = fullfile(folderPath, fname);
        
        % Attempt to parse X, Y, Z from the filename
        tokens = regexp(fname, pattern, 'tokens');
        if isempty(tokens)
            % Skip files that do not match the pattern
            continue;
        end
        
        % Convert captured strings into numeric values
        x = str2double(tokens{1}{1});
        y = str2double(tokens{1}{2});
        z = str2double(tokens{1}{3});
        
        % Load the .mat file (assumes variables named 'features' and 'timestamps')
        S = load(fpath, 'features', 'timestamps');
        if ~isfield(S, 'features') || ~isfield(S, 'timestamps')
            fprintf('Warning: "%s" does not contain "features" or "timestamps". Skipped.\n', fname);
            continue;
        end
        
        % Add an entry to our map for camera x
        if ~isKey(dataMap, x)
            dataMap(x) = {};  % create an empty cell array for storing rows
        end
        % Append a row: {y, z, featuresCell}
        dataMap(x) = [dataMap(x); {y, z, S.features, S.timestamps}];
    end
    
    % Prepare output struct
    camData = struct();
    camKeys = dataMap.keys;
    
    for i = 1:numel(camKeys)
        xVal = camKeys{i};              % camera index
        yzFeatures = dataMap(xVal);     % cell array of rows {y, z, features}
        
        % Sort rows first by y, then by z
        yzMatrix = cell2mat(yzFeatures(:, 1:2));  % Nx2 numeric array
        [~, sortIdx] = sortrows(yzMatrix, [1, 2]);
        yzFeatures = yzFeatures(sortIdx, :);
        
        % Concatenate features in sorted order
        % Each row's third element is a cell array: yzFeatures{j,3}
        % We'll do a horizontal concatenation of all these sub-cell-arrays
        allFeatures = {};
        for j = 1:size(yzFeatures, 1)

            subFeature = yzFeatures{j, 3}';
            subTimestamp = num2cell(yzFeatures{j, 4})'; % The 'features' cell array
            seq_len = length(subTimestamp);
            cam = cell(seq_len, 1); cam(:) = {xVal};
            vid = cell(seq_len, 1); vid(:) = {yzFeatures{j, 1}};
            seg = cell(seq_len, 1); seg(:) = {yzFeatures{j, 2}};
            % Concatenate horizontally (or adapt as needed)
            allFeatures = [allFeatures; subFeature, cam, vid, seg, subTimestamp];
        end
        
        % Store in the output struct under camData(xVal).features
        camField = sprintf('cam%d', xVal);  % e.g. 'cam1'
        camData.(camField) = allFeatures;
    end
end