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
            if ~isempty(s1) & ~isempty(s2)
                % Remove singletons from s1 before computing scores
                s1_filtered = removeSingletons(s1);
                data_results.homogeneity{i} = compute_homogeneity(s1_filtered, s2);
                data_results.split_score{i} = compute_split_score(s1_filtered, s2);
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
heatmap(hm);
figure;
heatmap(sp);

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

%% Helper Functions

function filtered_groups = removeSingletons(groups)
% REMOVESINGLETONS - Remove singleton groups (groups with only one person) from the input
%
% Input:
%   groups - Cell array where each cell contains a vector of person IDs
%
% Output:
%   filtered_groups - Cell array with singleton groups removed

    if isempty(groups)
        filtered_groups = {};
        return;
    end
    
    filtered_groups = {};
    for i = 1:length(groups)
        if length(groups{i}) > 1  % Keep only groups with more than one person
            filtered_groups{end+1} = groups{i};
        end
    end
end