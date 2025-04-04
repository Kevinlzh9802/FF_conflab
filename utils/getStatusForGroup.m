function statusGroup = getStatusForGroup(person, status, group)
% GETSTATUSFORGROUP Finds status values for indices where person == group values.
%
% Inputs:
%   - person: A vector containing person IDs.
%   - status: A vector of the same size as person containing status values.
%   - group: A 1xN cell array, where each cell contains an array of person IDs.
%
% Output:
%   - statusGroup: A 1xN cell array containing corresponding status values.
%     If a person ID in group is not found in person, it is assigned a value of -1000.
%
% Example Usage:
%   statusGroup = getStatusForGroup(person, status, group);

    % Define the special value for missing IDs
    missingValue = -1000;
    
    % Preallocate output cell array
    statusGroup = cell(size(group));

    % Loop through each cell in group
    for i = 1:numel(group)
        % Get the person IDs from the current group cell
        groupPersons = group{i};

        % Find indices where person matches groupPersons
        [found, idx] = ismember(groupPersons, person);
        
        % Initialize status array with missing value
        statusValues = ones(size(groupPersons)) * missingValue;
        
        % Assign valid status values where found
        statusValues(found) = status(idx(found));
        
        % Store in output cell array
        statusGroup{i} = statusValues;
    end
end
