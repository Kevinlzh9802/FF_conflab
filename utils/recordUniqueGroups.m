function result = recordUniqueGroups(T, g_name)

% Initialize map to collect unique groups with their timestamps and cams
group_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

for k = 1:height(T)
    current_groups = T.(g_name){k};      % a 1×n cell array of vectors
    current_time = T.concat_ts(k);       % timestamp
    current_cam  = T.Cam(k);             % camera ID

    for g = 1:length(current_groups)
        group_vec = sort(current_groups{g});
        group_key = mat2str(group_vec);

        entry = struct('timestamp', current_time, 'cam', current_cam);

        if isKey(group_map, group_key)
            group_map(group_key) = [group_map(group_key); entry];
        else
            group_map(group_key) = entry;
        end
    end
end

% Convert map to table
keys = group_map.keys;
num_entries = numel(keys);
group_list = cell(num_entries, 1);
timestamp_list = cell(num_entries, 1);
cam_list = cell(num_entries, 1);

for i = 1:num_entries
    group_list{i} = str2num(keys{i}); %#ok<ST2NM>
    entries = group_map(keys{i});
    timestamp_list{i} = [entries.timestamp]';
    cam_list{i} = unique([entries.cam]);
end

% Final output table
result = table(group_list, timestamp_list, cam_list, ...
    'VariableNames', {'participants', 'timestamps', 'Cam'});
end
