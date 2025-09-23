function [filtered_table, pairwise_diffs, analysis_results] = processWindowTable(window_table, aggregation_method)
% PROCESSWINDOWTABLE - Process window_table to filter rows and calculate pairwise differences
%
% Inputs:
%   window_table - Table with columns: id, Vid, time, length, speaking_all_time,
%                  detection, filtered_speakers, num_filtered_speakers, total_groups
%                  where num_filtered_speakers and total_groups are cell arrays
%   aggregation_method - String specifying the aggregation method used ('no_aggregation' or others)
%
% Outputs:
%   filtered_table - Filtered window_table with only rows where num_filtered_speakers
%                    is identical across all k values and >= 1
%   pairwise_diffs - Matrix of pairwise differences between total_groups columns
%                    (k x k matrix where pairwise_diffs(i,j) = sum(col_i - col_j))
%   analysis_results - Struct containing additional analysis results:
%                      - non_identical_rows: indices of rows where total_groups differ across columns
%                      - diff_distributions: cell array of (num_speakers - total_groups) for each column
%                      - column_names: names of the group detection columns
%                      - homogeneity_matrix: k×k matrix of average homogeneity scores between detection methods
%                      - split_score_matrix: k×k matrix of average split scores between detection methods
%                      - window_sizes: array of window sizes for non-identical rows
%                      - camera_ids: array of camera IDs for non-identical rows
%                      - is_distribution: boolean indicating if results are distribution-based

% Handle default parameter
if nargin < 2
    aggregation_method = 'closest_to_start';
end

% Determine if we're dealing with distribution-based results
is_distribution = strcmp(aggregation_method, 'no_aggregation');

% Step 1: Filter rows where num_filtered_speakers is identical across all k values and >= 1
valid_rows = [];

for i = 1:height(window_table)
    num_speakers_row = window_table.num_filtered_speakers{i};
    
    if is_distribution
        % For distribution-based results, num_speakers_row is a cell array
        % Check if all features have the same number of detections and all values >= 2
        if iscell(num_speakers_row) && ~isempty(num_speakers_row)
            % Get the length of each feature's detection array
            feature_lengths = cellfun(@length, num_speakers_row);
            
            % Check if all features have the same number of detections
            if isscalar(unique(feature_lengths))
                % Check if all detection results across all features are >= 2
                all_values = [];
                for feat_idx = 1:length(num_speakers_row)
                    all_values = [all_values, num_speakers_row{feat_idx}];
                end
                
                if all(all_values >= 2)
                    valid_rows = [valid_rows, i];
                end
            end
        end
    else
        % For non-distribution results, use the original logic
        % Check if all values are identical and >= 2
        if isscalar(unique(num_speakers_row)) && num_speakers_row(1) >= 2
            valid_rows = [valid_rows, i];
        end
    end
end

% Create filtered table
filtered_table = window_table(valid_rows, :);

if isempty(valid_rows)
    warning('No rows found where num_filtered_speakers is identical across all columns and >= 1');
    pairwise_diffs = [];
    return;
end

% Step 2: Calculate pairwise differences between total_groups columns
if is_distribution
    % For distribution-based results, total_groups is a cell array
    k = length(filtered_table.total_groups{1}); % Number of columns
    n = height(filtered_table); % Number of rows
    
    % For distribution-based results, we'll handle individual detection results
    % No need to convert to matrix format since we'll process each detection separately
else
    % For non-distribution results, use the original logic
    k = length(filtered_table.total_groups{1}); % Number of columns
    n = height(filtered_table); % Number of rows
    
    % Convert total_groups to matrix format
    total_groups_matrix = cell2mat(filtered_table.total_groups);
end

% Calculate pairwise differences
pairwise_diffs = zeros(k, k);
for i = 1:k
    for j = 1:k
        if i == j
            pairwise_diffs(i, j) = 0; % Diagonal elements are 0
        else
            if is_distribution
                % For distribution-based results, calculate differences across all individual detections
                all_diffs = [];
                for row_idx = 1:n
                    total_groups_i = filtered_table.total_groups{row_idx}{i};
                    total_groups_j = filtered_table.total_groups{row_idx}{j};
                    
                    if ~isempty(total_groups_i) && ~isempty(total_groups_j)
                        % Calculate differences for each detection position
                        for t = 1:length(total_groups_i)
                            all_diffs = [all_diffs, total_groups_i(t) - total_groups_j(t)];
                        end
                    end
                end
                pairwise_diffs(i, j) = sum(all_diffs);
            else
                % For non-distribution results, use the original logic
                pairwise_diffs(i, j) = sum(total_groups_matrix(:, i) - total_groups_matrix(:, j));
            end
        end
    end
end

% Display summary
fprintf('Processing complete:\n');
fprintf('- Original table: %d rows\n', height(window_table));
fprintf('- Filtered table: %d rows\n', height(filtered_table));
fprintf('- Number of group detection columns (k): %d\n', k);
fprintf('- Pairwise differences matrix size: %d x %d\n', size(pairwise_diffs));

% Additional analysis: Find rows where total_groups are not identical across k detections
fprintf('\n=== Additional Analysis ===\n');

% Step 1: Find rows where total_groups are not identical across k values
non_identical_rows = [];
for i = 1:height(filtered_table)
    total_groups_row = filtered_table.total_groups{i};
    
    if is_distribution
        % For distribution-based results, total_groups_row is a cell array
        if iscell(total_groups_row) && ~isempty(total_groups_row)
            % Check if any individual detection differs across features
            has_differences = false;
            
            % Get the number of detections (should be same for all features)
            num_detections = length(total_groups_row{1});
            
            for t = 1:num_detections
                % Get values for this detection position across all features
                detection_values = [];
                for feat_idx = 1:length(total_groups_row)
                    if t <= length(total_groups_row{feat_idx})
                        detection_values = [detection_values, total_groups_row{feat_idx}(t)];
                    end
                end
                
                % Check if values differ for this detection position
                if length(unique(detection_values)) > 1
                    has_differences = true;
                    break;
                end
            end
            
            if has_differences
                non_identical_rows = [non_identical_rows, i];
            end
        end
    else
        % For non-distribution results, use the original logic
        if length(unique(total_groups_row)) > 1  % Not all values are identical
            non_identical_rows = [non_identical_rows, i];
        end
    end
end

fprintf('- Rows with non-identical total_groups: %d out of %d (%.1f%%)\n', ...
    length(non_identical_rows), height(filtered_table), ...
    100 * length(non_identical_rows) / height(filtered_table));

% Analyze window distribution in non-identical rows
fprintf('- Window distribution analysis for non-identical rows:\n');

% Get window sizes and cameras for non-identical rows
window_sizes = [];
camera_ids = [];
for row_idx = non_identical_rows
    window_sizes = [window_sizes, filtered_table.length(row_idx)];
    camera_ids = [camera_ids, filtered_table.Cam(row_idx)];
end

% Window size distribution
unique_sizes = unique(window_sizes);
fprintf('  Window sizes:\n');
for size_val = unique_sizes
    count = sum(window_sizes == size_val);
    fprintf('    - Size %d: %d windows (%.1f%%)\n', size_val, count, ...
        100 * count / length(non_identical_rows));
end

% Camera distribution
unique_cameras = unique(camera_ids);
fprintf('  Camera distribution:\n');
for cam = unique_cameras
    count = sum(camera_ids == cam);
    fprintf('    - Camera %d: %d windows (%.1f%%)\n', cam, count, ...
        100 * count / length(non_identical_rows));
end

% Cross-tabulation of window sizes and cameras
fprintf('  Cross-tabulation (Window Size × Camera):\n');
for size_val = unique_sizes
    fprintf('    Size %d: ', size_val);
    for cam = unique_cameras
        count = sum(window_sizes == size_val & camera_ids == cam);
        if count > 0
            fprintf('Cam%d(%d) ', cam, count);
        end
    end
    fprintf('\n');
end

if ~isempty(non_identical_rows)
    % Step 2: Calculate distribution of (num_filtered_speakers - total_groups) for each column
    if is_distribution
        fprintf('- Analyzing distribution of (num_filtered_speakers - total_groups) for non-identical rows (distribution-based results):\n');
    else
        fprintf('- Analyzing distribution of (num_filtered_speakers - total_groups) for non-identical rows:\n');
    end
    
    % Get the difference distribution for each column
    diff_distributions = cell(k, 1);
    % Adjust based on your actual column names
    column_names = {'headRes', 'shoulderRes', 'hipRes', 'footRes'}; 
    
    % Collect all differences and unique values first
    all_differences = cell(k, 1);
    all_unique_diffs = [];
    
    for col_idx = 1:k
        differences = [];
        for row_idx = non_identical_rows
            if is_distribution
                % For distribution-based results, use individual values directly
                num_speakers_array = filtered_table.num_filtered_speakers{row_idx}{col_idx};
                total_groups_array = filtered_table.total_groups{row_idx}{col_idx};
                
                if ~isempty(num_speakers_array) && ~isempty(total_groups_array)
                    % Calculate differences for each timestamp individually
                    for t = 1:length(num_speakers_array)
                        diff = num_speakers_array(t) - total_groups_array(t);
                        differences = [differences, diff];
                    end
                end
            else
                % For non-distribution results, use the existing logic
                num_speakers = filtered_table.num_filtered_speakers{row_idx}(1); % Same across all columns
                total_groups = filtered_table.total_groups{row_idx}(col_idx);
                diff = num_speakers - total_groups;
                differences = [differences, diff];
            end
        end
        
        diff_distributions{col_idx} = differences;
        all_differences{col_idx} = differences;
        all_unique_diffs = [all_unique_diffs, differences];
        
        % Display statistics for this column
        if ~isempty(differences)
            fprintf('  Column %d (%s):\n', col_idx, column_names{min(col_idx, length(column_names))});
            fprintf('    - Mean: %.2f\n', mean(differences));
            fprintf('    - Std: %.2f\n', std(differences));
            fprintf('    - Min: %.2f\n', min(differences));
            fprintf('    - Max: %.2f\n', max(differences));
            
            if is_distribution
                fprintf('    - Note: Values are individual instances counted across all timestamps\n');
            end
            
            fprintf('    - Distribution: ');
            
            % Show frequency distribution (round to nearest integer for display)
            rounded_diffs = round(differences);
            unique_diffs = unique(rounded_diffs);
            for diff_val = unique_diffs
                count = sum(rounded_diffs == diff_val);
                fprintf('%d(%d) ', diff_val, count);
            end
            fprintf('\n');
        end
    end
    
    % Create combined bar plot with all columns
    if ~isempty(all_unique_diffs)
        all_unique_diffs = unique(all_unique_diffs);
        
        % Create matrix for grouped bar plot
        bar_data = zeros(length(all_unique_diffs), k);
        for col_idx = 1:k
            for diff_idx = 1:length(all_unique_diffs)
                bar_data(diff_idx, col_idx) = sum(all_differences{col_idx} == all_unique_diffs(diff_idx));
            end
        end
        
        % Create grouped bar plot
        figure('Name', 'Distribution Comparison Across All Columns');
        bar(all_unique_diffs, bar_data);
        
        % Customize the plot
        xlabel('Difference Values d_\alpha^w');
        ylabel('Window instance number');
        if is_distribution
            title('Distribution Comparison Across All Detection Methods (Individual Instance Counting)');
        else
            title('Distribution Comparison Across All Detection Methods');
        end
        legend(column_names(1:k), 'Location', 'best');
        grid on;
        
        % Adjust x-axis to accommodate grouped bars
        xlim([min(all_unique_diffs)-0.5, max(all_unique_diffs)+0.5]);
    end
    
    % Step 3: Create bubble plot for group size vs filtered speakers
    fprintf('- Creating bubble plot for group size vs filtered speakers:\n');
    create_bubble_plot(filtered_table, non_identical_rows, column_names, k, is_distribution);
    
    % Step 4: Calculate spatial scores (homogeneity and split score) for non-identical rows
    if is_distribution
        fprintf('- Calculating spatial scores for non-identical rows (distribution-based results):\n');
    else
        fprintf('- Calculating spatial scores for non-identical rows:\n');
    end
    [hm, sp] = calculate_spatial_scores(filtered_table, non_identical_rows, k, is_distribution);
    
    % Display spatial scores
    fprintf('  Homogeneity Matrix:\n');
    disp(hm);
    fprintf('  Split Score Matrix:\n');
    disp(sp);
    
    % Create heatmaps
    figure;
    heatmap(column_names, column_names, hm, 'Title', 'Homogeneity Scores');

    figure;
    heatmap(column_names, column_names, sp, 'Title', 'Split Scores');
    
    % Store results in output
    analysis_results.non_identical_rows = non_identical_rows;
    analysis_results.diff_distributions = diff_distributions;
    analysis_results.column_names = column_names;
    analysis_results.homogeneity_matrix = hm;
    analysis_results.split_score_matrix = sp;
    analysis_results.window_sizes = window_sizes;
    analysis_results.camera_ids = camera_ids;
    analysis_results.is_distribution = is_distribution;

    filtered_table = filtered_table(non_identical_rows, :);
    
else
    fprintf('- No rows found with non-identical total_groups.\n');
    analysis_results.non_identical_rows = [];
    analysis_results.diff_distributions = {};
    analysis_results.column_names = {};
    analysis_results.homogeneity_matrix = [];
    analysis_results.split_score_matrix = [];
    analysis_results.window_sizes = [];
    analysis_results.camera_ids = [];
    analysis_results.is_distribution = is_distribution;
end
end

%% Helper Functions

function create_bubble_plot(filtered_table, non_identical_rows, column_names, k, is_distribution)
    % CREATE_BUBBLE_PLOT - Creates a bubble plot showing group size vs filtered speakers
    % for each detection method
    
    figure('Name', 'Group Size vs Filtered Speakers Bubble Plot');
    
    % Define colors for different methods
    colors = lines(k);
    
    hold on;
    legend_entries = {};
    
    for col_idx = 1:k
        % Collect data for this method and count occurrences
        data_points = [];
        
        for row_idx = non_identical_rows
            % Get detection results for this method
            detection_result = filtered_table.detection{row_idx}{col_idx};
            if ~isempty(detection_result)
                if is_distribution
                    % For distribution-based results, process each timestamp
                    for t = 1:length(detection_result)
                        timestamp_detection = detection_result{t};
                        if ~isempty(timestamp_detection)
                            % For each group g in this timestamp, create a data point
                            for g = 1:length(timestamp_detection)
                                group = timestamp_detection{g};
                                
                                % X: Size of this specific group
                                group_size = length(group);
                                if group_size == 1
                                    continue;
                                end
                                
                                % Y: How many filtered speakers this specific group contains
                                filtered_speakers_in_group = filtered_table.filtered_speakers{row_idx}{col_idx}{t};
                                filtered_count = sum(ismember(group, filtered_speakers_in_group));
                                
                                % Store this data point (each group is a separate point)
                                data_points = [data_points; group_size, filtered_count];
                            end
                        end
                    end
                else
                    % For non-distribution results, use the existing logic
                    % For each group g, create a data point
                    for g = 1:length(detection_result)
                        group = detection_result{g};
                        
                        % X: Size of this specific group
                        group_size = length(group);
                        if group_size == 1
                            continue;
                        end
                        
                        % Y: How many filtered speakers this specific group contains
                        filtered_speakers_in_group = filtered_table.filtered_speakers{row_idx};
                        filtered_count = sum(ismember(group, filtered_speakers_in_group{col_idx}));
                        
                        % Store this data point (each group is a separate point)
                        data_points = [data_points; group_size, filtered_count];
                    end
                end
            end
        end
        
        if ~isempty(data_points)
            % Count occurrences of each (group_size, filtered_speaker_count) combination
            unique_combinations = unique(data_points, 'rows');
            bubble_sizes = [];
            
            for i = 1:size(unique_combinations, 1)
                group_size = unique_combinations(i, 1);
                filtered_count = unique_combinations(i, 2);
                
                % Count how many times this combination appears
                count = sum(all(data_points == [group_size, filtered_count], 2));
                bubble_sizes = [bubble_sizes, count];
            end
            
            % Arrange bubbles horizontally by column index to avoid overlap
            x_offset = (col_idx - 2.5) * 0.1; % Horizontal spacing between columns
            
            % Create scatter plot with adjusted x-positions and vertical oval markers
            scatter(unique_combinations(:, 1) + x_offset, unique_combinations(:, 2), ...
                bubble_sizes * 15, colors(col_idx, :), ...
                'Marker', 'o', 'LineWidth', 3);
            
            legend_entries{end+1} = column_names{col_idx};
        end
    end
    
    % Customize the plot
    xlabel('Detected Group Size');
    ylabel('Number of Simultaneous Speakers in Group');
    % title('Group Size vs Filtered Speakers Distribution');
    legend(legend_entries, 'Location', 'best');
    grid on;
    
    % Add some padding to axes and accommodate horizontal arrangement
    if ~isempty(data_points)
        max_group_size = max(data_points(:, 1));
        max_filtered_count = max(data_points(:, 2));
        xlim([1, max_group_size + (k-1)*0.3]); % Account for horizontal spacing
        ylim([-1, max_filtered_count + 1]);
    end
    
    % Add vertical grid lines to help distinguish columns
    ax = gca;
    ax.GridAlpha = 0.3;
    ax.MinorGridAlpha = 0.1;
    
    hold off;
end

function [hm, sp] = calculate_spatial_scores(filtered_table, non_identical_rows, k, is_distribution)
    % CALCULATE_SPATIAL_SCORES - Calculates homogeneity and split score matrices
    
    % Initialize spatial scores matrices
    hm = zeros(k, k);  % homogeneity matrix
    sp = zeros(k, k);  % split score matrix
    
    % Calculate pairwise spatial scores between all k detection methods
    for k1 = 1:k
        for k2 = 1:k
            homogeneity_scores = [];
            split_scores = [];
            
            for row_idx = non_identical_rows
                % Get detection results for the two methods being compared
                s1 = filtered_table.detection{row_idx}{k1};  % Method k1
                s2 = filtered_table.detection{row_idx}{k2};  % Method k2
                
                if ~isempty(s1) && ~isempty(s2)
                    if is_distribution
                        % For distribution-based results, calculate spatial scores across all timestamps
                        timestamp_homogeneity = [];
                        timestamp_split_scores = [];
                        
                        % Ensure both s1 and s2 have the same number of timestamps
                        min_timestamps = min(length(s1), length(s2));
                        
                        for t = 1:min_timestamps
                            if ~isempty(s1{t}) && ~isempty(s2{t})
                                timestamp_homogeneity = [timestamp_homogeneity, compute_homogeneity(s1{t}, s2{t})];
                                timestamp_split_scores = [timestamp_split_scores, compute_split_score(s1{t}, s2{t})];
                            end
                        end
                        
                        % Take mean of spatial scores across timestamps
                        if ~isempty(timestamp_homogeneity)
                            homogeneity = mean(timestamp_homogeneity);
                            split_score = mean(timestamp_split_scores);
                            
                            homogeneity_scores = [homogeneity_scores, homogeneity];
                            split_scores = [split_scores, split_score];
                        end
                    else
                        % For non-distribution results, calculate spatial scores directly
                        homogeneity = compute_homogeneity(s1, s2);
                        split_score = compute_split_score(s1, s2);
                        
                        homogeneity_scores = [homogeneity_scores, homogeneity];
                        split_scores = [split_scores, split_score];
                    end
                end
            end
            
            % Calculate average scores
            if ~isempty(homogeneity_scores)
                hm(k1, k2) = mean(homogeneity_scores);
                sp(k1, k2) = mean(split_scores);
            else
                hm(k1, k2) = NaN;
                sp(k1, k2) = NaN;
            end
        end
    end
end

% function filtered_groups = removeSingletons(groups)
% % REMOVESINGLETONS - Remove singleton groups (groups with only one person) from the input
% %
% % Input:
% %   groups - Cell array where each cell contains a vector of person IDs
% %
% % Output:
% %   filtered_groups - Cell array with singleton groups removed

%     if isempty(groups)
%         filtered_groups = {};
%         return;
%     end
    
%     filtered_groups = {};
%     for i = 1:length(groups)
%         if length(groups{i}) > 1  % Keep only groups with more than one person
%             filtered_groups{end+1} = groups{i};
%         end
%     end
% end
