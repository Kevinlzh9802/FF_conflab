function plotSkeletonOnImage(ax, img, keypoints, kp_set, use_real)
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
    % fig = figure(figHandle); % Keep previous contents
    % ax = axes(fig);

    % Display the image
    if ~use_real
        imshow(img, 'Parent', ax); 
        hold(ax, "on");
    end
    
    % Get image dimensions
    [imgHeight, imgWidth, ~] = size(img);
    
    % Get field names (each representing a different person)
    % colNames = fieldnames(keypoints);
    sz_kp = size(keypoints, 1);
    
    % Define a set of distinct colors
    colors = lines(sz_kp); % Generates distinguishable colors
    
    % Labels for body parts
    labels = {"Head", "Shoulder", "Hip", "Foot"};

    lengthScale = 0.03 * imgWidth; % Scale perpendicular length
    
    % Loop through each person's keypoints
    for k = 1:sz_kp
        % kp = keypoints.(colNames{k});
        kp = reshape(keypoints(k, 2:end), [2,8])';
        person_id = num2str(keypoints(k,1));
        if ~isempty(kp)
            % plot certain kps
            kp_filter = [];
            for kpi = kp_set
                kp_filter = [kp_filter, 2*kpi-1:2*kpi];
            end
            % Convert keypoints from ratio to pixel coordinates
            if use_real
                pixelCoordsX = kp(kp_filter,1); % Do not scale, real world coordinates in cm
                pixelCoordsY = kp(kp_filter,2);
            else
                pixelCoordsX = kp(kp_filter,1) * imgWidth; % Scale x-coordinates
                pixelCoordsY = kp(kp_filter,2) * imgHeight; % Scale y-coordinates
            end

            
            % Plot keypoints with a unique color
            scatter(ax, pixelCoordsX, pixelCoordsY, 50, colors(k, :), 'filled');
            hold(ax, "on");
            
            % Draw lines between each pair of keypoints
            text_id_done = false;
            for i = 1:length(kp_set)
                labelIdx = kp_set(i);
                plot_i = 2*i - 1;
                if ~ismember(labelIdx, kp_set)
                    continue;
                end
                
                x1 = pixelCoordsX(plot_i); x2 = pixelCoordsX(plot_i+1);
                y1 = pixelCoordsY(plot_i); y2 = pixelCoordsY(plot_i+1);
                
                % Skip if points are invalid (NaN or too close)
                all_valid = ~(isnan(x1) || isnan(y1) || isnan(x2) || isnan(y2) || (x1 == x2 && y1 == y2));
                
                % Draw main segment
                if all_valid
                    plot(ax, [x1, x2], [y1, y2], '-', 'Color', colors(k, :), 'LineWidth', 2); 
                    hold(ax, "on");
                end
                
                % Compute middle point
                midX = (x1 + x2) / 2;
                midY = (y1 + y2) / 2;
                
                % labelIdx = (i + 1) / 2;
                % Label left and right positions
                dx = x2 - x1;
                dy = y2 - y1;
                
                % Special case for head (first two rows)
                if labelIdx == 1
                    % Label Head and Nose
                    text(ax, x1, y1, 'H', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
                    text(ax, x2, y2, 'N', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
                    if ~text_id_done
                        text(ax, x1 - 10, y1 - 10, person_id, 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                        text_id_done = true;
                    end
                    if (dx ~= 0 || dy ~= 0) && all_valid
                        x_diff = lengthScale * (dx / sqrt(dx^2 + dy^2));
                        y_diff = lengthScale * (dy / sqrt(dx^2 + dy^2));
                        % Draw arrow in the same direction as L->R
                        quiver(ax, midX, midY, x_diff, y_diff, 0, 'Color', colors(k, :), 'LineWidth', 1.5, 'MaxHeadSize', 1);
                        
                    end
                else
                    % Compute half perpendicular segment (counterclockwise 90 degrees from L->R)
                    text(ax, x1, y1, 'L', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
                    text(ax, x2, y2, 'R', 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
                    if ~text_id_done
                        text(ax, x1 - 10, y1 - 10, person_id, 'Color', colors(k, :), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                        text_id_done = true;
                    end
                    if (dx ~= 0 || dy ~= 0) && all_valid
                        % Perpendicular direction (-dy, dx), counterclockwise half segment
                        x_diff = lengthScale * (dy / sqrt(dx^2 + dy^2));
                        y_diff = -lengthScale * (dx / sqrt(dx^2 + dy^2));
                        % Draw perpendicular segment with arrows
                        quiver(ax, midX, midY, x_diff, y_diff, 0, 'Color', colors(k, :), 'LineWidth', 1.5, 'MaxHeadSize', 1);
                    end
                end
                % Add labels if within predefined body parts (1-2: Head, 3-4: Shoulder, etc.)
                
                if labelIdx <= length(labels)
                    text(ax, midX, midY, labels{labelIdx}, 'Color', colors(k, :), 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                end
            end
        end
        axis equal;
    end
    
    hold(ax, "off");
end
