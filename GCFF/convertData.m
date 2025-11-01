% clear variables; close all;
parent_dir = '../data/export'; mkdir(parent_dir); 

%% key points
% for clue = ["head", "shoulder", "hip", "foot"]
%     file_name = sprintf("../data/%s.mat", clue);
%     load(file_name, 'all_data');
%     write_feature_data(all_data, clue, parent_dir);
% end

%% speaking status
load('../data/speaking_status.mat', 'speaking_status');
exportSpeaking(speaking_status);

%% frames
% load('../data/frames.mat', 'frames');
% exportFrames(frames)
% exportFramesArray(frames, '../data/export/frames.mat')

%% functions
function write_feature_data(T, clue, parent_dir)
    outdir = fullfile(parent_dir, clue);
    mkdir(outdir);
    mkdir(fullfile(outdir, 'features'));

    N = height(T);
    row_id = (1:N).';
    metadata = table(row_id, T.Cam, T.Vid, T.Seg, T.Timestamp, ...
        'VariableNames', {'row_id','Cam','Vid','Seg','Timestamp'});
    writetable(metadata, fullfile(outdir,'metadata.csv'));

    % features: write each n×48 matrix to its own CSV
    for i = 1:N
        Fi = T.Features{i};                 % numeric [n×48], n may vary
        writematrix(Fi, fullfile(outdir,'features', sprintf('f_%06d.csv', i)));
    end

    % GT: JSON Lines (list of lists)
    fid = fopen(fullfile(outdir,'gt.jsonl'),'w');
    for i = 1:N
        GTi = T.GT{i};                      % k×1 cell, each cell numeric row vector
        GT_lists = cellfun(@(v) reshape(v,1,[]), GTi, 'UniformOutput', false);
        obj = struct('row_id', i, 'GT', {GT_lists});
        fprintf(fid, '%s\n', jsonencode(obj));
    end
    fclose(fid);
end

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

function exportFrames(frames, outDir)
% exportFrames Save frame images from a MATLAB table to PNG files.
%   exportFrames(frames, outDir)
%   - frames: table with columns {FrameData, Cam, Vid, Seg, Timestamp}
%             where each FrameData row is an image array (e.g., 540x960x3 uint8)
%   - outDir: output directory (default: '../data/export/frames')
%
% Files are saved as: frame_CamVidSeg_Timestamp.png
% Example: frame_235_1120.png for Cam=2, Vid=3, Seg=5, Timestamp=1120.

if nargin < 2 || isempty(outDir)
    outDir = fullfile('..','data','export','frames');
end

if ~exist(outDir,'dir')
    mkdir(outDir);
end

assert(istable(frames), 'frames must be a table');
reqVars = {'FrameData','Cam','Vid','Seg','Timestamp'};
for v = reqVars
    assert(ismember(v{1}, frames.Properties.VariableNames), ...
        'frames table must contain variable %s', v{1});
end

n = height(frames);
for i = 1:n
    % Extract image data
    try
        img = frames.FrameData{i};  % works for cell or content
    catch
        img = frames.FrameData{i,1};
    end
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end

    % Extract metadata
    cam = frames.Cam(i);
    vid = frames.Vid(i);
    seg = frames.Seg(i);
    ts  = frames.Timestamp(i);
    % Unwrap cells, datetimes, etc.
    if iscell(cam), cam = cam{1}; end
    if iscell(vid), vid = vid{1}; end
    if iscell(seg), seg = seg{1}; end
    if iscell(ts),  ts  = ts{1};  end
    if isdatetime(ts)
        ts = posixtime(ts); % fallback to numeric
    end
    cam = double(cam); vid = double(vid); seg = double(seg);
    ts  = double(ts);  ts = round(ts);

    % Build filename and save
    fname = sprintf('frame_%d%d%d_%d.png', cam, vid, seg, ts);
    try
        imwrite(img, fullfile(outDir, fname));
    catch
        img_empty = uint8(zeros(540, 960, 3));
        imwrite(img_empty, fullfile(outDir, fname));
    end
end

fprintf('Exported %d frames to %s\n', n, outDir);

end

function exportFramesArray(frames, outMat)
% exportFramesArray Save all frame images into a single 4D array in a v7.3 MAT.
%   exportFramesArray(frames, outMat)
%   - frames: table with columns {FrameData, Cam, Vid, Seg, Timestamp}
%             FrameData is (H x W x 3) uint8 per row
%   - outMat: path to output MAT file (recommended .mat, will be v7.3)
%
% The MAT will contain:
%   - frames4d: uint8 [H x W x 3 x N]
%   - Cam, Vid, Seg: double [N x 1]
%   - Timestamp: double [N x 1]
%
% Notes:
% - For large N this will exceed 2GB; v7.3 is required. We write incrementally
%   using matfile to avoid holding the full array in memory.

assert(istable(frames), 'frames must be a table');
reqVars = {'FrameData','Cam','Vid','Seg','Timestamp'};
for v = reqVars
    assert(ismember(v{1}, frames.Properties.VariableNames), ...
        'frames table must contain variable %s', v{1});
end

N = height(frames);
% Infer H, W from first non-empty frame
idx = find(~cellfun(@isempty, frames.FrameData), 1);
if isempty(idx)
    error('No FrameData found');
end
img0 = frames.FrameData{idx};
[H,W,C] = size(img0);
assert(C==3, 'Expected 3 channels');

% Prepare matfile (v7.3)
mf = matfile(outMat, 'Writable', true);
% Pre-size datasets on disk
mf.frames4d = zeros(H, W, 3, N, 'uint8');
mf.Cam = zeros(N,1);
mf.Vid = zeros(N,1);
mf.Seg = zeros(N,1);
mf.Timestamp = zeros(N,1);

for i = 1:N
    img = frames.FrameData{i};
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end
    try
        mf.frames4d(:,:,:,i) = img;
    catch
        mf.frames4d(:,:,:,i) = uint8(zeros(H, W, 3));
        disp("warning: empty image array at position " + i);
    end
    mf.Cam(i,1) = double(frames.Cam(i));
    mf.Vid(i,1) = double(frames.Vid(i));
    mf.Seg(i,1) = double(frames.Seg(i));
    ts = frames.Timestamp(i);
    if iscell(ts), ts = ts{1}; end
    if isdatetime(ts), ts = posixtime(ts); end
    mf.Timestamp(i,1) = double(ts);
end

fprintf('Saved %d frames into %s (frames4d: %dx%dx%dx%d)\n', N, outMat, H,W,3,N);

end
