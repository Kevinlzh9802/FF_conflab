function filteredTable = filterTable(dataTable, camValues, vidValues, segValues)
% FILTERTABLEBYCAMVIDSEG Filters rows from a table based on cam, vid, and seg.
%
% Inputs:
%   - dataTable: The input table containing columns 'cam', 'vid', and 'seg'.
%   - camValues: A vector of camera IDs to filter, or 'all' for no filtering.
%   - vidValues: A vector of video IDs to filter, or 'all' for no filtering.
%   - segValues: A vector of segment IDs to filter, or 'all' for no filtering.
%
% Output:
%   - filteredTable: A table containing only rows matching the specified cam, vid, and seg values.
%
% Example Usage:
%   filteredData = filterTableByCamVidSeg(myTable, [1, 2], 'all', [1, 3]);

    % Check if 'all' is selected for any filter and set logical masks accordingly
    if isequal(camValues, 'all')
        camFilter = true(size(dataTable.Cam));
    else
        camValues = camValues(:);
        camFilter = ismember(dataTable.Cam, camValues);
    end
    
    if isequal(vidValues, 'all')
        vidFilter = true(size(dataTable.Vid));
    else
        vidValues = vidValues(:);
        vidFilter = ismember(dataTable.Vid, vidValues);
    end
    
    if isequal(segValues, 'all')
        segFilter = true(size(dataTable.Seg));
    else
        segValues = segValues(:);
        segFilter = ismember(dataTable.Seg, segValues);
    end
    
    % Apply all filters together
    finalFilter = camFilter & vidFilter & segFilter;
    
    % Extract filtered rows
    filteredTable = dataTable(finalFilter, :);
end



% frame_seg2 = convertCellArrayToTable(frame_seg2.allFrames);
% frame_seg3 = convertCellArrayToTable(frame_seg3.allFrames);
% frames = [frame_seg2; frame_seg3];
% save("frames.mat", "frames", '-v7.3');