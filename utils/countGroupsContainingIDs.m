function num_groups_per_row = countGroupsContainingIDs(group_column, specified_groups)
% COUNTGROUPSCONTAININGIDS counts how many groups in each row of group_column
% contain any of the specified IDs, under the condition that every specified
% group has at least one member appearing in the row's union of IDs.
%
% Inputs:
%   - group_column: cell array (n x 1), each cell contains a {k x 1} cell array of ID vectors
%   - specified_groups: cell array of vectors, each vector is a group of specified IDs
%
% Output:
%   - num_groups_per_row: array with count per row, or 0 if any specified group is missing

    num_groups_per_row = zeros(size(group_column));

    for i = 1:length(group_column)
        groups = group_column{i};  % n×1 cell, each element is a vector of IDs

        if isempty(groups)
            num_groups_per_row(i) = 0;
            continue;
        end

        all_ids_in_row = [];
        for g = 1:length(groups)
            all_ids_in_row = [all_ids_in_row; groups{g}(:)];
        end
        all_ids_in_row = unique(all_ids_in_row);

        % Check validity of each specified group
        all_groups_valid = true;
        for k = 1:length(specified_groups)
            if isempty(intersect(specified_groups{k}, all_ids_in_row))
                all_groups_valid = false;
                break;
            end
        end

        if ~all_groups_valid
            num_groups_per_row(i) = 0;
            continue;
        end

        all_ids_spec = [];
        for g = 1:length(specified_groups)
            all_ids_spec = [all_ids_spec; specified_groups{g}(:)];
        end
        all_ids_spec = unique(all_ids_spec);

        % Count how many actual groups contain any specified ID
        count = 0;
        for j = 1:length(groups)
            group = groups{j};
            if any(ismember(group, all_ids_spec))
                count = count + 1;
            end
        end

        num_groups_per_row(i) = count;
    end
end
