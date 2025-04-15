function [sp, cf] = readSpeakingStatus(sp_status, vid, seg, n)
% READNTHROWFROMCSV Reads the nth row from a CSV file named "vidX_segY.csv".
%
% Inputs:
%   - x: Video index (integer, used in "vidX")
%   - y: Segment index (integer, used in "segY")
%   - n: Row number to extract (integer)
%
% Output:
%   - rowData: Data from the nth row of the specified CSV file
%
% Example Usage:
%   row = readNthRowFromCSV(3, 2, 5);

    % Read from mat file
    fn = "vid" + vid + "_seg" + seg;
    sp = sp_status.speaking.(fn);
    cf = sp_status.confidence.(fn);
    % % Construct the filename based on x and y
    % speaking = sprintf(['/home/zonghuan/tudelft/projects/datasets/conflab/' ...
    %     'annotations/actions/speaking_status/processed/speaking/vid%d_seg%d.csv'], x, y);
    % confidence = sprintf(['/home/zonghuan/tudelft/projects/datasets/conflab/' ...
    %     'annotations/actions/speaking_status/processed/confidence/vid%d_seg%d.csv'], x, y);
    % 
    % % Check if file exists
    % if ~isfile(speaking)
    %     error('File %s does not exist.', speaking);
    % end
    % if ~isfile(confidence)
    %     error('File %s does not exist.', confidence);
    % end
    % 
    % % Read the CSV file
    % sp = readmatrix(speaking);
    % cf = readmatrix(confidence);

    % Validate n
    if n < 1 || n > size(sp, 1)
        error('Row index n is out of bounds for file %s.', sp);
    end
    if n < 1 || n > size(cf, 1)
        error('Row index n is out of bounds for file %s.', cf);
    end

    % Return the nth row
    sp = sp(n, :);
    cf = cf(n, :);
end
