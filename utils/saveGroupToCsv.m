function a = saveGroupToCsv(T)
all_formations = [];
for i = 1:height(T)
    row_groups = T.formations{i};  % e.g., {[1 2 3], [4 5]}
    for j = 1:length(row_groups)
        group = row_groups{j};
        participant_str = strjoin(string(group), ' ');
        all_formations(end+1,:) = {participant_str}; %#ok<AGROW>
    end
end

% Convert to table
f_table = cell2table(all_formations, 'VariableNames', {'participants'});

% Optional: save to CSV (then load in Python)
writetable(f_table, 'f_formations.csv')
a = 0;
end