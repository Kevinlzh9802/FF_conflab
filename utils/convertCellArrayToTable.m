function outputTable = convertCellArrayToTable(cellArray)
% CONVERTCELLARRAYTOTABLE Converts a cell array into a structured table.
%
% The function:
%   - Keeps column 1 the same.
%   - Splits column 2 into "cam", "vid", and "seg" by extracting numbers.
%   - Copies column 3 into a new column 5.
%
% Input:
%   - cellArray: A cell array with format shown in the input image.
%
% Output:
%   - outputTable: A table with structured columns.
%
% Example Usage:
%   outputTable = convertCellArrayToTable(myCellArray);

    numRows = size(cellArray, 1);
    
    % Preallocate extracted numerical values
    cam = zeros(numRows, 1);
    vid = zeros(numRows, 1);
    seg = zeros(numRows, 1);
    
    for i = 1:numRows
        % Extract numbers from the string in column 2 (e.g., 'cam2_vid2_seg8')
        tokens = regexp(cellArray{i, 2}, 'cam(\d+)_vid(\d+)_seg(\d+)', 'tokens');
        
        if ~isempty(tokens)
            cam(i) = str2double(tokens{1}{1});
            vid(i) = str2double(tokens{1}{2});
            seg(i) = str2double(tokens{1}{3});
        else
            warning('Row %d: Could not parse "%s"', i, cellArray{i, 2});
        end
    end
    
    % Create a table with the new structured format
    outputTable = table(cellArray(:, 1), cam, vid, seg, cellArray(:, 3), ...
                        'VariableNames', {'FrameData', 'Cam', 'Vid', 'Seg', 'Timestamp'});

end
