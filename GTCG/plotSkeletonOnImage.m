function plotSkeletonOnImage(figHandle, img, keypoints)
% PLOTSKELETONONIMAGE Plots multiple sets of skeleton keypoints on an image, using different colors for each person.
% Draws lines between each pair of keypoints (e.g., left & right hand, foot, etc.), and adds perpendicular segments with labels.
%
% Inputs:
%   - figHandle: Handle to the figure where the image should be displayed.
%   - img: Image matrix.
%   - keypoints: Struct where each field contains an Nx2 matrix of keypoint ratios for different persons.
%
% Example Usage:
%   fig = figure;
%   img = imread('example.jpg');
%   keypoints.person1 = [0.5 0.6; 0.7 0.8; 0.2 0.3; 0.4 0.5; 0.3 0.4; 0.5 0.6; 0.2 0.5; 0.6 0.7];
%   plotSkeletonOnImage(fig, img, keypoints);

    % Activate the given figure handle
    figure(figHandle); % Keep previous contents
    
    % Display the image
    imshow(img); hold on;
    
    % Get image dimensions
    [imgHeight, imgWidth, ~] = size(img);
    
    % Get field names (each representing a different person)
    colNames = fieldnames(keypoints);
    
    % Define a set of distinct colors
    colors = lines(length(colNames)); % Generates distinguishable colors
    
    % Labels for body parts
    labels = {"Head", "Shoulder", "Hip", "Foot"};
    
    % Loop through each person's keypoints
    for k = 1:length(colNames)
        kp = keypoints.(colNames{k});
        if ~isempty(kp)
            % Convert keypoints from ratio to pixel coordinates
            pixelCoordsX = kp(:,1) * imgWidth; % Scale x-coordinates
            pixelCoordsY = kp(:,2) * imgHeight; % Scale y-coordinates
            
            % Plot keypoints with a unique color
            scatter(pixelCoordsX, pixelCoordsY, 50, colors(k, :), 'filled');
            
            % Draw lines between each pair of keypoints
            for i = 1:2:size(kp,1)-1
                x1 = pixelCoordsX(i); x2 = pixelCoordsX(i+1);
                y1 = pixelCoordsY(i); y2 = pixelCoordsY(i+1);
                
                % Skip if points are invalid (NaN or too close)
                if isnan(x1) || isnan(y1) || isnan(x2) || isnan(y2) || (x1 == x2 && y1 == y2)
                    continue;
                end
                
                % Draw main segment
                plot([x1, x2], [y1, y2], '-', 'Color', colors(k, :), 'LineWidth', 2);
                
                % Compute middle point
                midX = (x1 + x2) / 2;
                midY = (y1 + y2) / 2;
                
                % Label left and right positions
                text(x1, y1, 'L', 'Color', colors(k, :), 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
                text(x2, y2, 'R', 'Color', colors(k, :), 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
                
                % Compute half perpendicular segment (counterclockwise 90 degrees from L->R)
                dx = x2 - x1;
                dy = y2 - y1;
                lengthScale = 0.02 * imgWidth; % Scale perpendicular length
                
                if dx ~= 0 || dy ~= 0
                    % Perpendicular direction (-dy, dx), counterclockwise half segment
                    perpX = midX - lengthScale * (dy / sqrt(dx^2 + dy^2));
                    perpY = midY + lengthScale * (dx / sqrt(dx^2 + dy^2));

                    % Draw perpendicular segment with arrows
                    quiver(midX, midY,  - perpX + midX, -perpY + midY, 0, 'Color', colors(k, :), 'LineWidth', 1.5, 'MaxHeadSize', 1);

                    % Add labels if within predefined body parts (1-2: Head, 3-4: Shoulder, etc.)
                    labelIdx = (i + 1) / 2;
                    if labelIdx <= length(labels)
                        text(midX, midY, labels{labelIdx}, 'Color', colors(k, :), 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                    end
                end
            end
        end
    end
    
    hold off;
end