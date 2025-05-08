%% Function to Display Frustums and Image in Subplots
function plotFrustumsWithImage(personData, frustum, img, disp_info)
% PLOTFRUSTUMSWITHIMAGE Displays frustums and an image in a subplot.
% 
% Inputs:
%   - personData: nx4 matrix containing person data
%   - frustum: Struct with frustum parameters
%   - img: Image matrix to display
%   - cellArray: A cell array to display at the bottom
%
% Example Usage:
%   img = imread('example.jpg');
%   plotFrustumsWithImage(personData, frustum, img, {{1,2}, {3,4,5}});

    fig = figure;
    subplot(1,2,1);
    plotFrustums(personData, frustum, fig);
    title('Frustum View');

    subplot(1,2,2);
    % plotSkeletonOnImage(fig, img, disp_info.kp);
    title('Image View');

    % Display cell array as text
    text(0.5, -0.1, ['GT: ', formatCellArray(disp_info.GT)], 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 18);
    text(0.5, -0.2, ['Detection: ', formatCellArray(disp_info.detection)], 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 18);
    text(0.5, -0.3, ['Speaking: ', formatCellArray(disp_info.speaking)], 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 18);
    text(0.5, -0.4, ['Confidence: ', formatCellArray(disp_info.confidence)], 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 18);
end



%% Function to format cell array into string
function outputStr = formatCellArray(cellArray)
% FORMATCELLARRAY Converts a nested cell array into a formatted string.
%
% Example:
%   cellArray = {{[32 15 2 17]}, {[22 11]}, {[25 10]}, {[3]}};
%   outputStr = formatCellArray(cellArray);
%
% Output:
%   '{{32,15,2,17}}, {{22,11}}, {{25,10}}, {{3}}'

    % Convert each numeric array inside the cell to a string with commas
    formattedCells = cellfun(@(x) mat2str(x, 2), cellArray, 'UniformOutput', false);

    % Wrap each formatted cell with curly braces
    outputStr = strjoin(formattedCells, ', ');
end
