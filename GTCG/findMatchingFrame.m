function frame = findMatchingFrame(table1, table2, n)
% FINDMATCHINGFRAME Finds a matching frame in table2 based on table1's nth row.
%
% Inputs:
%   - table1: The first table containing Features, Cam, Vid, Seg, Timestamp, GT.
%   - table2: The second table containing FrameData, Cam, Vid, Seg, Timestamp.
%   - n: The row index in table1 to search for.
%
% Output:
%   - frame: The corresponding frame data from table2 if a match is found; 
%            otherwise, returns an empty array.
%
% Example Usage:
%   frame = findMatchingFrame(table1, table2, 10);

    % Ensure n is within valid bounds
    if n < 1 || n > height(table1)
        error('Row index n is out of bounds.');
    end

    % Extract Cam, Vid, Seg from table1 at row n
    camValue = table1.Cam(n);
    vidValue = table1.Vid(n);
    segValue = table1.Seg(n);
    T = table1.Timestamp(n);

    % Search for the corresponding row in table2
    % matchIdx = find(table2.Cam == camValue & ...
    %                 table2.Vid == vidValue & ...
    %                 table2.Seg == segValue, 1);
    % Search for the corresponding row in table2
    matchIdx = find(table2.Cam == table1.Cam(n) & ...
                    table2.Vid == table1.Vid(n) & ...
                    table2.Seg == table1.Seg(n) & ...
                    cell2mat(table2.Timestamp) == T , 1);

    % If a match is found, return the corresponding FrameData
    if ~isempty(matchIdx)
        frame = table2.FrameData{matchIdx}; % Extract the frame from the table
    else
        frame = []; % Return empty if no match is found
        warning('No matching row found in table2.');
    end
end
