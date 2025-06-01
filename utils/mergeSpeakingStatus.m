function merged = mergeSpeakingStatus(A,B)
% Extract IDs and data
idsA = A(1, :); dataA = A(2:end, :);
idsB = B(1, :); dataB = B(2:end, :);

% Union of all IDs
all_ids = unique([idsA, idsB]);

% Initialize merged matrices with NaNs
mergedA = NaN(size(dataA, 1), numel(all_ids));
mergedB = NaN(size(dataB, 1), numel(all_ids));

% Map A data into mergedA
[~, locA] = ismember(idsA, all_ids);
mergedA(:, locA) = dataA;

% Map B data into mergedB
[~, locB] = ismember(idsB, all_ids);
mergedB(:, locB) = dataB;

% Final merged matrix
merged = [all_ids; mergedA; mergedB];
end