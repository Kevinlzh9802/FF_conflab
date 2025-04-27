function [sp, cf] = readSpeakingStatus(sp_status, vid, seg, n, window)
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
fn = "vid" + vid + "_seg" + seg;
sp = sp_status.speaking.(fn);
cf = sp_status.confidence.(fn);

if window == 1
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

elseif window > 1
    % Validate n
    w_start = n - round(window * 0.5);
    w_end = w_start + window - 1;
    try
        sp = mean(sp(w_start:w_end, :));
        cf = mean(cf(w_start:w_end, :));
    catch
        sp = -1000;
        cf = -1000;
    end
end
end
