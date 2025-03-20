function plotFrustums(personData, frustum)
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

    figure; hold on; axis equal;
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
