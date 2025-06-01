close all;

for vid=2:3
for cam=2:2:8
    mask = data_results.Cam == cam & data_results.Vid == vid;
    rows = data_results(mask, ["Seg","Timestamp"]);
    data_results(mask, ["Seg","Timestamp", "concat_ts"]) = add_merged_column(rows);
    % c = 9;
end
end


% concat_segs.cam2 = {[2,9], [3,2,3,4,5]};
% concat_segs.cam4 = {[2,9], [3,1], [3,3,4,5]};
% concat_segs.cam6 = {[2,9], [3,2,3,4,5]};
% concat_segs.cam8 = {[2,9], [3,3,4,5]};
% 
% cams = 2:2:8;
% concat_id = 1;
% data_results.concat_ts = zeros(height(data_results), 1);
% for cam=cams
%     c_segs = concat_segs.("cam" + cam);
%     for k=1:length(c_segs)
%         vid = c_segs{k}(1);
%         segs = c_segs{k}(2:end);
%         data_results = merge_segments(data_results, cam, vid, segs, ...
%             concat_id, speaking_status.speaking);
%         concat_id = concat_id + 1;
%     end
% end

function T = merge_segments(T, cam, vid, segs, concat_id)
%MERGE_SEGMENTS Merges timestamps for selected video segments
%
%   T = merge_segments(T, vid, segs)
%
%   Inputs:
%       T    - Input table with fields: Vid, Seg, Timestamp
%       vid  - Video number to filter on (scalar)
%       segs - Vector of segment numbers to merge (e.g., [1 3 4])
%
%   Output:
%       T    - Modified table with a new column 'concat_ts' for matching rows

    % Ensure 'concat_ts' column exists
    if ~ismember('concat_ts', T.Properties.VariableNames)
        T.concat_ts = repmat({[]}, height(T), 1);
    end

    % Step 1: filter relevant rows
    mask = T.Cam == cam & T.Vid == vid & ismember(T.Seg, segs);
    rows = find(mask);

    % Step 2: collect timestamps in the order of `segs`
    ts_all = [];
    
    if isscalar(segs)
        if vid == 2
            seq_id = segs - 8;
        elseif vid == 3
            seq_id = segs - 1;
        end

        seg_rows = find(T.Cam == cam & T.Vid == vid & T.Seg == segs);
        prev_last = 0;
        ts_all = T.Timestamp(seg_rows);
    elseif length(segs) > 1
        prev_last = 0;
        for s_ind=1:length(segs)
            s = segs(s_ind);
            if vid == 2
                seq_id = s - 8;
            elseif vid == 3
                seq_id = s - 1;
            end

            seg_rows = find(T.Cam == cam & T.Vid == vid & T.Seg == s);
            
            modified_ts = round(prev_last + T.Timestamp(seg_rows));
            ts_all = [ts_all; modified_ts];

            prev_last = ts_all(end) + 59;
        end
    end
    % for s = segs
    % 
    % 
    %     ts_all = [ts_all; T.Timestamp(seg_rows)];
    %     % for r = seg_rows'
    %     %     ts_all = [ts_all, T.Timestamp(r)];
    %     % end
    % end

    % Step 3: assign the combined timestamp to all matching rows
    % T.concat_ts = zeros();
    T.concat_ts(rows) = ts_all;
    T.concat_id(rows) = concat_id;
    % for r = rows'
    %     T.concat_ts{r} = ts_all;
    % end
end

function T = add_merged_column(T)
    % Get unique IDs in order
    ids = unique(T.Seg, 'stable');

    % Initialize merged frame counter
    offset = 0;
    T.concat_ts = NaN(height(T),1);  % preallocate

    for i = 1:length(ids)
        idx = T.Seg == ids(i);
        frames = T.Timestamp(idx);
        % nFrames = length(frames);

        % Generate merged frame sequence for this ID
        merged_frames = offset + frames;

        % Assign to table
        T.concat_ts(idx) = merged_frames';

        % Update offset
        offset = merged_frames(end);
    end
end