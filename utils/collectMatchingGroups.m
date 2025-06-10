function combinedGroups = collectMatchingGroups(vector, ts, vid, cam, ...
    results, feat_name, speaking_status, window_len)
% COLLECT_MATCHING_GROUPS filters and concatenates groups from results table by matching time values
%
% Inputs:
%   - vector: numeric vector of IDs to match
%   - ts: time values to match against 'concat_ts' column in 'results'
%   - vid: video ID to match
%   - cam: camera ID to match
%   - results: table with columns 'concat_ts', 'Vid', 'Cam', and feature column (e.g. "GazeRes")
%   - feat_name: feature prefix name, used to extract 'feat_name + "Res"' column from results
%   - speaking_status: matrix with rows as time indices and columns as person speaking states
%   - window_len: odd integer specifying the length of the time window to average speaking status
%
% Output:
%   - combinedGroups: 3×k cell array with:
%       {1,k} - matched groups
%       {2,k} - original group speaking status
%       {3,k} - speaking status of matched groups

    combinedGroups = cell(3, 0);  % Initialize 3xk output cell
    feat_res = feat_name + "Res";
    speaking_status = speaking_status{vid};
    matchedRows = ismember(results.concat_ts, ts) & results.Vid == vid & results.Cam == cam;
    matchedIndices = find(matchedRows);

    % First row of speaking_status contains ID->column mapping
    id_list = speaking_status(1, :);
    speaking_data = speaking_status(2:end, :);  % actual data (exclude header)
    max_t = size(speaking_data, 1);
    half_win = floor(window_len / 2);

    for i = 1:length(matchedIndices)
        idx = matchedIndices(i);
        t = results.concat_ts(idx);

        if iscell(results.(feat_res))
            groups = results.(feat_res){idx};

            % Step 1: matched groups
            matched = filterGroupByMembers(vector, groups);

            % Step 2: original speaking status (average over window centered at t, columns in vector)
            [~, col_inds] = ismember(vector, id_list);
            t_start = max(1, t - half_win);
            t_end = min(max_t, t + half_win);
            orig_status = mean(speaking_data(t_start:t_end, col_inds), 1);

            % Step 3: split speaking status (average over window, columns in each matched group)
            split_status = cell(1, length(matched));
            for g = 1:length(matched)
                ids = matched{g};
                [~, group_cols] = ismember(ids, id_list);
                split_status{g} = mean(speaking_data(t_start:t_end, group_cols), 1);
            end

            % Store into combinedGroups
            combinedGroups{1, end+1} = matched;
            combinedGroups{2, end}   = orig_status;
            combinedGroups{3, end}   = split_status;
        end
    end
end

