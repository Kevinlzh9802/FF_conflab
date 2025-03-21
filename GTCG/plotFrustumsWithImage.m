%% Function to Display Frustums and Image in Subplots
function plotFrustumsWithImage(personData, frustum, img, cellArray)
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
    imshow(img);
    title('Image View');

    % Display cell array as text
    % text(0.5, -0.1, formatCellArray(cellArray), 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', 12);
end

function plotFrustums(personData, frustum, figHandle)
% PLOTFRUSTUMS Plots frustums for persons and highlights their intersections.
%
% Inputs:
%   - personData: nx4 matrix, where:
%       - Column 1: Person ID
%       - Column 2: x-coordinate
%       - Column 3: y-coordinate
%       - Column 4: Orientation in radians
%   - frustum: Struct with fields:
%       - length: Frustum radius
%       - aperture: Frustum angle in degrees
%
% Example Usage:
%   personData = [1, 0, 0, pi/4; 2, 2, 1, pi/2];
%   frustum.length = 3;
%   frustum.aperture = 60; % Degrees
%   plotFrustums(personData, frustum);

    figure(figHandle); hold on; axis equal;
    title('Frustum Visualization and Intersections');
    xlabel('X'); ylabel('Y');
    set(gca, 'YDir', 'reverse'); % Flip Y-axis

    numPeople = size(personData, 1);
    colors = lines(numPeople); % Distinct colors for each person

    % Convert frustum aperture from degrees to radians
    frustum.aperture = deg2rad(frustum.aperture);

    % Store frustum shapes
    frustumShapes = cell(numPeople, 1);

    for i = 1:numPeople
        % Extract person data
        id = personData(i, 1);
        x = personData(i, 2);
        y = personData(i, 3);
        theta = personData(i, 4);

        % Compute frustum sector
        frustumShapes{i} = computeFrustum(x, y, theta, frustum.length, frustum.aperture);

        % Plot frustum using patch
        plotColor = colors(i, :);
        plotFrustumShape(frustumShapes{i}, plotColor, 0.3);

        % Display person ID near center of frustum
        text(x - frustum.length / 20 * cos(theta), y - frustum.length / 20 * sin(theta), ...
            num2str(id), 'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end

    % Check for intersections
    for i = 1:numPeople-1
        for j = i+1:numPeople
            intersectionShape = intersect(frustumShapes{i}, frustumShapes{j});
            if ~isempty(intersectionShape.Vertices)
                plotFrustumShape(intersectionShape, [1, 0, 0], 0.5); % Red highlight
            end
        end
    end

    hold off;
end

%% Function to Compute Frustum Sector as a Polyshape
function frustumShape = computeFrustum(x, y, theta, length, aperture)
    numPoints = 30; % Resolution of the frustum boundary
    halfAngle = aperture / 2;

    % Compute boundary angles
    angles = linspace(theta - halfAngle, theta + halfAngle, numPoints);

    % Compute sector points
    xBoundary = x + length * cos(angles);
    yBoundary = y + length * sin(angles);

    % Construct polygon (starting from person location)
    xPoly = [x, xBoundary, x];
    yPoly = [y, yBoundary, y];

    frustumShape = polyshape(xPoly, yPoly);
end

%% Function to Plot a Frustum Shape with Transparency
function plotFrustumShape(shape, color, alphaVal)
    plot(shape, 'FaceColor', color, 'FaceAlpha', alphaVal, 'EdgeColor', 'none');
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
    formattedCells = cellfun(@(x) ['{' strjoin(string(x), ',') '}'], cellArray, 'UniformOutput', false);

    % Wrap each formatted cell with curly braces
    outputStr = ['{' strjoin(formattedCells, ', ') '}'];
end



