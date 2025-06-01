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
        found = findGroupCovers(target, B);
        if ~isempty(found)
            a_count = a_count + 1;
        end
    end

    % Step 4: Check how many groups in B are exactly covered by >1 groups in A
    b_count = 0;
    for i = 1:numel(B)
        target = B{i};
        found = findGroupCovers(target, A);
        if ~isempty(found)
            b_count = b_count + 1;
        end
    end

    result = [a_count, b_count];
end

function covers = findGroupCovers(target, groupSet)
% Find all combinations of >1 groups from groupSet whose union == target

    n = numel(groupSet);
    covers = {};

    for k = 2:n  % combinations of 2 or more groups
        combos = nchoosek(1:n, k);
        for i = 1:size(combos,1)
            union_set = unique([groupSet{combos(i,:)}]);
            if isequal(sort(union_set), sort(target))
                covers{end+1} = combos(i,:);
            end
        end
    end
end
