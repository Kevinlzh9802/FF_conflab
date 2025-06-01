function result = recordUniqueGroups(T, g_name)

% Initialize map to collect unique groups and timestamps
group_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

for k = 1:height(T)
    current_groups = T.(g_name){k};        % a 1×n cell array of vectors
    current_time = T.concat_ts(k);       % timestamp corresponding to this row

    for g = 1:length(current_groups)
        group_vec = sort(current_groups{g});        % sort to ensure consistent key
        group_key = mat2str(group_vec);             % convert to string key

        if isKey(group_map, group_key)
            group_map(group_key) = [group_map(group_key); current_time];
        else
            group_map(group_key) = current_time;
        end
    end
end

% Convert map to table
keys = group_map.keys;
num_entries = numel(keys);
group_list = cell(num_entries, 1);
timestamp_list = cell(num_entries, 1);

for i = 1:num_entries
    group_list{i} = str2num(keys{i});  %#ok<ST2NM>
    timestamp_list{i} = group_map(keys{i});
end

% Final output table
result = table(group_list, timestamp_list, 'VariableNames', {'participants', 'timestamps'});
end