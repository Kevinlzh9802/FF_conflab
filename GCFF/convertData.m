clear variables; close all;
parent_dir = '../data/export'; mkdir(parent_dir); 
%% key points
% for clue = ["head", "shoulder", "hip", "foot"]
%     file_name = sprintf("../data/%s.mat", clue);
%     load(file_name, 'all_data');
%     write_feature_data(all_data, clue, parent_dir);
% end
% function write_feature_data(T, clue, parent_dir)
%     outdir = fullfile(parent_dir, clue);
%     mkdir(outdir);
%     mkdir(fullfile(outdir, 'features'));
% 
%     N = height(T);
%     row_id = (1:N).';
%     metadata = table(row_id, T.Cam, T.Vid, T.Seg, T.Timestamp, ...
%         'VariableNames', {'row_id','Cam','Vid','Seg','Timestamp'});
%     writetable(metadata, fullfile(outdir,'metadata.csv'));
% 
%     % features: write each n×48 matrix to its own CSV
%     for i = 1:N
%         Fi = T.Features{i};                 % numeric [n×48], n may vary
%         writematrix(Fi, fullfile(outdir,'features', sprintf('f_%06d.csv', i)));
%     end
% 
%     % GT: JSON Lines (list of lists)
%     fid = fopen(fullfile(outdir,'gt.jsonl'),'w');
%     for i = 1:N
%         GTi = T.GT{i};                      % k×1 cell, each cell numeric row vector
%         GT_lists = cellfun(@(v) reshape(v,1,[]), GTi, 'UniformOutput', false);
%         obj = struct('row_id', i, 'GT', {GT_lists});
%         fprintf(fid, '%s\n', jsonencode(obj));
%     end
%     fclose(fid);
% end

%% speaking status
load('../data/speaking_status.mat', 'speaking_status');

exportSpeaking(speaking_status);

function exportSpeaking(speaking_status, outFile)
% convertData Export speaking status structs to a Python-friendly .mat
%   convertData(speaking_status, outFile)
%   - speaking_status: struct with fields 'speaking' and 'confidence'. Each
%     of those contains fields like 'vid2_seg8', 'vid3_seg1', ... that map to
%     numeric matrices [T x N].
%   - outFile: path to output .mat file (default: '../data/export/speaking_status_py.mat')
%
% The saved .mat will contain a single struct variable 'speaking_status' with
% two sub-structs 'speaking' and 'confidence' preserving field names and data
% arrays, so it can be loaded from Python (e.g., via scipy.io.loadmat) and
% passed to utils.speaking.read_speaking_status.

if nargin < 2 || isempty(outFile)
    outFile = fullfile('..','data','export','speaking_status_py.mat');
end

% Basic validation
assert(isstruct(speaking_status), 'speaking_status must be a struct');
assert(isfield(speaking_status,'speaking'), 'speaking_status must have field ''speaking''');
assert(isfield(speaking_status,'confidence'), 'speaking_status must have field ''confidence''');

S = struct();
S.speaking = struct();
S.confidence = struct();

% Copy all fields from speaking_status.speaking
sp_fields = fieldnames(speaking_status.speaking);
for i = 1:numel(sp_fields)
    fn = sp_fields{i};
    S.speaking.(fn) = speaking_status.speaking.(fn);
end

% Copy all fields from speaking_status.confidence
cf_fields = fieldnames(speaking_status.confidence);
for i = 1:numel(cf_fields)
    fn = cf_fields{i};
    S.confidence.(fn) = speaking_status.confidence.(fn);
end

% Also store an index of available keys for convenience
S.index = struct();
S.index.speaking_keys = sp_fields;
S.index.confidence_keys = cf_fields;

% Ensure output directory exists
[outDir,~,~] = fileparts(outFile);
if ~isempty(outDir) && ~exist(outDir,'dir')
    mkdir(outDir);
end

% Save as MATLAB v7 (SciPy-friendly)
speaking_status = S; %#ok<NASGU>
save(outFile, 'speaking_status', '-v7');

fprintf('Saved speaking status to %s\n', outFile);
fprintf('  speaking keys: %d, confidence keys: %d\n', numel(sp_fields), numel(cf_fields));

end

