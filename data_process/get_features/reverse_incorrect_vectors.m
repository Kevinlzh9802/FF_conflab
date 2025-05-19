function V_out = reverse_incorrect_vectors(V)
% V: 4x2 matrix of 2D vectors (each row is a vector)
% Returns V_out: 4x2 matrix with incorrect vectors reversed if needed

V_out = V;

% Find non-NaN vectors
valid_idx = find(~any(isnan(V), 2));
valid_vectors = V(valid_idx, :);
n = size(valid_vectors, 1);

if n <= 2
    return; % do nothing
end

% Compute pairwise dot products
dot_signs = sign(valid_vectors * valid_vectors');
dot_signs = tril(dot_signs, -1); % use lower triangle only to avoid duplicate pairs

% Count agreement per vector
agree_counts = sum(dot_signs == 1, 2);
disagree_counts = sum(dot_signs == -1, 2);

if n == 3
    % Look for 2 agreeing, 1 disagreeing
    for i = 1:3
        others = setdiff(1:3, i);
        dots = valid_vectors(i,:) * valid_vectors(others,:)';
        if all(dots < 0)
            V_out(valid_idx(i), :) = -V_out(valid_idx(i), :);
            return;
        end
    end

elseif n == 4
    % Check for 3 agreeing, 1 disagreeing
    for i = 1:4
        others = setdiff(1:4, i);
        dots = valid_vectors(i,:) * valid_vectors(others,:)';
        if all(dots < 0)
            V_out(valid_idx(i), :) = -V_out(valid_idx(i), :);
            return;
        end
    end
end
end