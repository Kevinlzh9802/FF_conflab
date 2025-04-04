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

    lengthScale = 0.03 * imgWidth; % Scale perpendicular length
    
    % Loop through each person's keypoints
    for k = 1:length(colNames)
        kp = keypoints.(colNames{k});
        person_id = colNames{k}(2:end);
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
                all_valid = ~(isnan(x1) || isnan(y1) || isnan(x2) || isnan(y2) || (x1 == x2 && y1 == y2));
                
                % Draw main segment
                if all_valid
                    plot([x1, x2], [y1, y2], '-', 'Color', colors(k, :), 'LineWidth', 2);
                end
                
                % Compute middle point
                midX = (x1 + x2) / 2;
                midY = (y1 + y2) / 2;
                
                % labelIdx = (i + 1) / 2;
                % Label left and right positions
                dx = x2 - x1;
                dy = y2 - y1;
                
                % Special case for head (first two rows)
                if i == 1
                    % Label Head and Nose
                    text(x1, y1, 'H', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
                    text(x2, y2, 'N', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
                    text(x1 - 10, y1 - 10, person_id, 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                    if (dx ~= 0 || dy ~= 0) && all_valid
                        x_diff = lengthScale * (dx / sqrt(dx^2 + dy^2));
                        y_diff = lengthScale * (dy / sqrt(dx^2 + dy^2));
                        % Draw arrow in the same direction as L->R
                        quiver(midX, midY, x_diff, y_diff, 0, 'Color', colors(k, :), 'LineWidth', 1.5, 'MaxHeadSize', 1);
                        
                    end
                else
                    % Compute half perpendicular segment (counterclockwise 90 degrees from L->R)
                    text(x1, y1, 'L', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
                    text(x2, y2, 'R', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
                    
                    if (dx ~= 0 || dy ~= 0) && all_valid
                        % Perpendicular direction (-dy, dx), counterclockwise half segment
                        x_diff = lengthScale * (dy / sqrt(dx^2 + dy^2));
                        y_diff = -lengthScale * (dx / sqrt(dx^2 + dy^2));
                        % Draw perpendicular segment with arrows
                        quiver(midX, midY, x_diff, y_diff, 0, 'Color', colors(k, :), 'LineWidth', 1.5, 'MaxHeadSize', 1);
                    end
                end
                % Add labels if within predefined body parts (1-2: Head, 3-4: Shoulder, etc.)
                labelIdx = (i + 1) / 2;
                if labelIdx <= length(labels)
                    text(midX, midY, labels{labelIdx}, 'Color', colors(k, :), 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                end
            end
        end
    end
    
    hold off;
end
