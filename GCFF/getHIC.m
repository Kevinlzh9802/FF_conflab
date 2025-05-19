used_data.HIC = cell(height(used_data), 1);
for k=1:height(used_data)
    % a = computeHICMatrix(used_data.GT{k}, used_data.headRes{k});
    used_data.HIC{k} = computeHICMatrix(used_data.GT{k}, used_data.headRes{k});
end
