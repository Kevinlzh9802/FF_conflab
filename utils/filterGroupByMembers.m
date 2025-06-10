function result = filterGroupByMembers(vector, groups)
% FILTER_GROUPS_BY_MEMBERS returns groups that have any overlap with the input vector
%
% Inputs:
%   - vector: a numeric vector of IDs
%   - groups: 1×m cell array, each cell is a numeric array (group of IDs)
%
% Output:
%   - result: cell array of groups that share at least one element with 'vector'

    result = {};  % initialize output
    for i = 1:numel(groups)
        group = groups{i};
        if any(ismember(group, vector))
            result{end+1} = group; %#ok<AGROW>
        end
    end
end