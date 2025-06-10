function [unique_vals, counts] = uniqueCellGroups(cell_array)
    % Convert each entry to a string for hashing
    n = numel(cell_array);
    str_reprs = cell(n, 1);
    for i = 1:n
        group = cell_array{i};
        group_sorted = cellfun(@sort, group, 'UniformOutput', false); % sort each vector
        str_parts = cellfun(@(v) mat2str(v), group_sorted, 'UniformOutput', false);
        str_reprs{i} = strjoin(str_parts, '|');  % join with separator
    end

    % Find unique string representations and their counts
    [unique_strs, ~, ic] = unique(str_reprs);
    counts_vec = accumarray(ic, 1);

    % Convert back to original structure
    unique_vals = cell(numel(unique_strs), 1);
    for i = 1:numel(unique_strs)
        parts = strsplit(unique_strs{i}, '|');
        unique_vals{i} = cellfun(@(s) str2num(s), parts, 'UniformOutput', false); %#ok<ST2NM>
    end

    counts = counts_vec;
end