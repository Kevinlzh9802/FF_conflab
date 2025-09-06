function [filtered_table, pairwise_diffs, analysis_results] = processWindowTable(window_table)
% PROCESSWINDOWTABLE - Process window_table to filter rows and calculate pairwise differences
%
% Inputs:
%   window_table - Table with columns: id, Vid, time, length, speaking_all_time,
%                  detection, filtered_speakers, num_filtered_speakers, total_groups
%                  where num_filtered_speakers and total_groups are cell arrays
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

% Step 1: Filter rows where num_filtered_speakers is identical across all k values and >= 1
valid_rows = [];

for i = 1:height(window_table)
    num_speakers_row = window_table.num_filtered_speakers{i};
    
    % Check if all values are identical and >= 1
    if isscalar(unique(num_speakers_row)) && num_speakers_row(1) >= 2
    % if num_speakers_row(1) >= 2
        valid_rows = [valid_rows, i];
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
k = length(filtered_table.total_groups{1}); % Number of columns
n = height(filtered_table); % Number of rows

% Convert total_groups to matrix format
total_groups_matrix = cell2mat(filtered_table.total_groups);

% Calculate pairwise differences
pairwise_diffs = zeros(k, k);
for i = 1:k
    for j = 1:k
        if i == j
            pairwise_diffs(i, j) = 0; % Diagonal elements are 0
        else
            % Calculate sum(col_i - col_j)
            pairwise_diffs(i, j) = sum(total_groups_matrix(:, i) - total_groups_matrix(:, j));
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
    if length(unique(total_groups_row)) > 1  % Not all values are identical
        non_identical_rows = [non_identical_rows, i];
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
    fprintf('- Analyzing distribution of (num_filtered_speakers - total_groups) for non-identical rows:\n');
    
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
            num_speakers = filtered_table.num_filtered_speakers{row_idx}(1); % Same across all columns
            total_groups = filtered_table.total_groups{row_idx}(col_idx);
            diff = num_speakers - total_groups;
            differences = [differences, diff];
        end
        
        diff_distributions{col_idx} = differences;
        all_differences{col_idx} = differences;
        all_unique_diffs = [all_unique_diffs, differences];
        
        % Display statistics for this column
        if ~isempty(differences)
            fprintf('  Column %d (%s):\n', col_idx, column_names{min(col_idx, length(column_names))});
            fprintf('    - Mean: %.2f\n', mean(differences));
            fprintf('    - Std: %.2f\n', std(differences));
            fprintf('    - Min: %d\n', min(differences));
            fprintf('    - Max: %d\n', max(differences));
            fprintf('    - Distribution: ');
            
            % Show frequency distribution
            unique_diffs = unique(differences);
            for diff_val = unique_diffs
                count = sum(differences == diff_val);
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
        xlabel('Difference Values (|S_w| - n_\alpha^w)');
        ylabel('Window instance number');
        title('Distribution Comparison Across All Detection Methods');
        legend(column_names(1:k), 'Location', 'best');
        grid on;
        
        % Adjust x-axis to accommodate grouped bars
        xlim([min(all_unique_diffs)-0.5, max(all_unique_diffs)+0.5]);
    end
    
    % Step 3: Create bubble plot for group size vs filtered speakers
    fprintf('- Creating bubble plot for group size vs filtered speakers:\n');
    create_bubble_plot(filtered_table, non_identical_rows, column_names, k);
    
    % Step 4: Calculate spatial scores (homogeneity and split score) for non-identical rows
    fprintf('- Calculating spatial scores for non-identical rows:\n');
    [hm, sp] = calculate_spatial_scores(filtered_table, non_identical_rows, k);
    
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
end
end

%% Helper Functions

function create_bubble_plot(filtered_table, non_identical_rows, column_names, k)
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

function [hm, sp] = calculate_spatial_scores(filtered_table, non_identical_rows, k)
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
                    % Calculate spatial scores
                    homogeneity = compute_homogeneity(s1, s2);
                    split_score = compute_split_score(s1, s2);
                    
                    homogeneity_scores = [homogeneity_scores, homogeneity];
                    split_scores = [split_scores, split_score];
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
