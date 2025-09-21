% Homogeneity and split score
hm = zeros(4);
sp = zeros(4);
for k1=1:4
    for k2=1:4
        g1 = 0;
        g2 = 0;
        for i=1:height(data_results)
            r1 = clues(k1) + "Res";
            r2 = clues(k2) + "Res";
            s1 = data_results.(r1){i};
            s2 = data_results.(r2){i};
            % if ~isempty(s1) & ~isempty(s2)
            if allCluesValid(data_results, clues, i)
                data_results.homogeneity{i} = compute_homogeneity(s1, s2);
                data_results.split_score{i} = compute_split_score(s1, s2);
                g1 = g1 + 1;
                g2 = g2 + 1;
            else
                data_results.homogeneity{i} = [];
                data_results.split_score{i} = [];
            end

        end
        % r1,r2
        hm(k1, k2) = sum(cell2mat(data_results.homogeneity)) / g1;
        sp(k1, k2) = sum(cell2mat(data_results.split_score)) / g1;
    end
end
figure; heatmap(hm);
figure; heatmap(sp);

% Calculate group size distribution for each clue under allCluesValid conditions
fprintf('\n=== Group Size Distribution Analysis ===\n');
clues = ["head", "shoulder", "hip", "foot"];

for clue_idx = 1:length(clues)
    clue = clues(clue_idx);
    clue_res = clue + "Res";
    
    % Collect all group sizes for this clue under allCluesValid conditions
    group_sizes = [];
    
    for i = 1:height(data_results)
        if allCluesValid(data_results, clues, i)
            groups = data_results.(clue_res){i};
            if ~isempty(groups)
                for g = 1:length(groups)
                    group_size = length(groups{g});
                    if group_size >= 2  % Only count groups with size >= 2
                        group_sizes = [group_sizes; group_size];
                    end
                end
            end
        end
    end
    
    % Calculate distribution
    if ~isempty(group_sizes)
        unique_sizes = unique(group_sizes);
        counts = zeros(size(unique_sizes));
        for j = 1:length(unique_sizes)
            counts(j) = sum(group_sizes == unique_sizes(j));
        end
        
        % Display results
        fprintf('\n%s Clue:\n', clue);
        fprintf('  Total groups (size >= 2): %d\n', sum(counts));
        fprintf('  Mean group size: %.2f\n', mean(group_sizes));
        fprintf('  Std group size: %.2f\n', std(group_sizes));
        fprintf('  Size countings:\n');
        for j = 1:length(unique_sizes)
            fprintf('    Size %d: %d groups (%.1f%%)\n', ...
                unique_sizes(j), counts(j), 100*counts(j)/sum(counts));
        end
    else
        fprintf('\n%s Clue: No groups found with size >= 2\n', clue);
    end
end

fprintf('\n=== Summary ===\n');
for clue_idx = 1:length(clues)
    clue = clues(clue_idx);
    clue_res = clue + "Res";
    
    % Recalculate for summary
    group_sizes = [];
    for i = 1:height(data_results)
        if allCluesValid(data_results, clues, i)
            groups = data_results.(clue_res){i};
            if ~isempty(groups)
                for g = 1:length(groups)
                    group_size = length(groups{g});
                    if group_size >= 2
                        group_sizes = [group_sizes; group_size];
                    end
                end
            end
        end
    end
    
    if ~isempty(group_sizes)
        fprintf('%s: %d groups, mean size %.2f\n', ...
            clue, length(group_sizes), mean(group_sizes));
    else
        fprintf('%s: No groups found\n', clue);
    end
end

for sp_name = clues
    if sp_name ~= base_clue
        col_name = sp_name + "Split";
        for k=1:height(formations)
            formations.(col_name){k} = collectMatchingGroups(formations.participants{k}, ...
                formations.longest_ts{k}, formations.Vid(k), ...
                formations.Cam(k), data_results, sp_name, sp_merged, 61);
        end
    end
end

function allCluesValid = allCluesValid(data_results, clues, row_num)
    allCluesValid = true;
    for i=1:length(clues)
        if isempty(data_results.(clues(i) + "Res"){row_num})
            allCluesValid = false;
        end
    end
end