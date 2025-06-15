function result = setCoverage(A, B)
% SETCOVERAGE computes how many groups in A are exactly covered
% by >1 groups in B, and vice versa.

    % Step 1: Flatten both sets to get the universe of elements
    A_flat = vertcat(A{:});
    B_flat = vertcat(B{:});
    universe = unique([A_flat; B_flat]);

    % Step 2: Pad A and B so all elements in universe are covered
    missing_in_B = setdiff(A_flat, B_flat);
    missing_in_A = setdiff(B_flat, A_flat);

    % Add singleton groups for missing elements
    for x = missing_in_B'
        B{end+1} = x;
    end
    for x = missing_in_A'
        A{end+1} = x;
    end

    % Convert all groups to sorted row vectors for comparison
    A = cellfun(@(c) sort(c(:)'), A, 'UniformOutput', false);
    B = cellfun(@(c) sort(c(:)'), B, 'UniformOutput', false);

    % Step 3: Check how many groups in A are exactly covered by >1 groups in B
    a_count = 0;
    for i = 1:numel(A)
        target = A{i};
        found = findGroupCoverScore(target, B);
        a_count = a_count + found;
        % if ~isempty(found)
        % 
        % end
    end

    % Step 4: Check how many groups in B are exactly covered by >1 groups in A
    b_count = 0;
    for i = 1:numel(B)
        target = B{i};
        found = findGroupCoverScore(target, A);
        b_count = b_count + found;
        % if ~isempty(found)
        % 
        % end
    end

    result = [a_count, b_count];
end

function score = findGroupCoverScore(target, groupSet)
% Given a target vector and a cell array of group vectors (disjoint),
%   - Returns 0 if target is equal to or a subset of any group in groupSet
%   - Returns 1 if some combination's union == target
%   - Else, finds S1 (maximal subset), S2 (minimal superset), and returns e^(s-1)
%     where s = min(|S2|/|target|, |target|/|S1|)

    n = numel(groupSet);
    target = sort(target);
    targetSet = unique(target);

    % New rule: Early exit if target is equal to or subset of any group
    for j = 1:n
        group_j = unique(groupSet{j});
        if isequal(targetSet, group_j) || all(ismember(targetSet, group_j))
            score = 0;
            return;
        end
    end

    % Early exit for exact cover (multiple groups)
    for k = 2:n
        combos = nchoosek(1:n, k);
        for i = 1:size(combos,1)
            unionSet = sort([groupSet{combos(i,:)}]);
            if isequal(unionSet, targetSet)
                score = 1;
                return;
            end
        end
    end

    % If not found, look for closest S1 (subset) and S2 (superset)
    bestSubset = [];
    bestSubsetLen = 0;
    bestSuperset = [];
    bestSupersetLen = inf;

    for k = 1:n
        combos = nchoosek(1:n, k);
        for i = 1:size(combos,1)
            unionSet = unique([groupSet{combos(i,:)}]);
            if all(ismember(unionSet, targetSet)) % subset
                if numel(unionSet) > bestSubsetLen
                    bestSubset = unionSet;
                    bestSubsetLen = numel(unionSet);
                end
            end
            if all(ismember(targetSet, unionSet)) % superset
                if numel(unionSet) < bestSupersetLen
                    bestSuperset = unionSet;
                    bestSupersetLen = numel(unionSet);
                end
            end
        end
    end

    % Calculate s and score
    if isempty(bestSubset), S1 = 0; else, S1 = numel(bestSubset); end
    if isinf(bestSupersetLen), S2 = inf; else, S2 = bestSupersetLen; end
    tlen = numel(targetSet);

    if S1 == 0 || isinf(S2)
        score = 0; % No valid subset or superset found
    else
        s = min(S2/tlen, tlen/S1);
        score = exp(s-1);
    end
end


