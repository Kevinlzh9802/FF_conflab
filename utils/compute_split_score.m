function score = compute_split_score(G1, G2)
    % Ensure all groups within G1 and G2 are column vectors
    for i = 1:length(G1)
        if isrow(G1{i})
            G1{i} = G1{i}';
        end
    end
    for i = 1:length(G2)
        if isrow(G2{i})
            G2{i} = G2{i}';
        end
    end
    
    % Step 1: union of all elements
    all_elems = unique([vertcat(G1{:}); vertcat(G2{:})]);
    N = numel(all_elems);

    % Pad missing elements in G1 and G2 with singleton groups
    G1_elems = vertcat(G1{:});
    G2_elems = vertcat(G2{:});
    missing_G1 = setdiff(all_elems, G1_elems);
    missing_G2 = setdiff(all_elems, G2_elems);
    for i = 1:numel(missing_G1)
        G1{end+1} = missing_G1(i);
    end
    for i = 1:numel(missing_G2)
        G2{end+1} = missing_G2(i);
    end

    % Step 2: compute split score
    total = 0;
    for i = 1:numel(G1)
        A = G1{i};  % group in G1
        if isempty(A)
            continue;
        end
        subgroup_count = 0;
        for j = 1:numel(G2)
            B = G2{j};  % group in G2
            overlap = intersect(A, B);
            if numel(overlap) > 1  % only count non-trivial subgroups
                subgroup_count = subgroup_count + 1;
            end
        end
        % Contribution: size * max(0, k(A) - 1)
        total = total + numel(A) * max(0, subgroup_count - 1);
    end

    score = total / length(G1);
end
