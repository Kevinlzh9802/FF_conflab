function T_out = filterAndConcatTable(T, keys)
% T: Input table with columns 'Cam', 'Seg', 'Vid'
% keys: Cell array of strings, each representing a 3-digit identifier

    T_out = T([],:);  % Initialize empty table with same structure

    for i = 1:numel(keys)
        key = keys{i};
        if length(key) ~= 3 || any(~isstrprop(key, 'digit'))
            error('Each key must be a 3-digit numeric string.');
        end

        cam = str2double(key(1));
        vid = str2double(key(2));
        seg = str2double(key(3));

        % Filter matching rows
        mask = T.Cam == cam & T.Vid == vid & T.Seg == seg;
        T_out = [T_out; T(mask, :)];  % Concatenate
    end
end