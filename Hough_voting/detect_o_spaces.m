function o_spaces = detect_o_spaces(Atilde, AL, threshold)
% Detect o-space peaks
o_spaces = [];
[rows, cols] = find(Atilde > threshold);

for idx = 1:length(rows)
    o_spaces(end+1).center = [cols(idx), rows(idx)];
    o_spaces(end).members = AL{rows(idx), cols(idx)};
end
end
