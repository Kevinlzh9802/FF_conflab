function h = compute_homogeneity(G1, G2)
    % Combine all unique elements across both groupings
    all_elems = unique([vertcat(G1{:}); vertcat(G2{:})]);
    N = numel(all_elems);

    % Create element to index map
    elem_map = containers.Map(all_elems, 1:N);

    % Align/pad groups so all elements appear in both G1 and G2
    G1_flat = vertcat(G1{:});
    G2_flat = vertcat(G2{:});
    missing_in_G1 = setdiff(all_elems, G1_flat);
    missing_in_G2 = setdiff(all_elems, G2_flat);

    % Pad G1 and G2 with singleton groups
    for i = 1:numel(missing_in_G1)
        G1{end+1} = missing_in_G1(i);
    end
    for i = 1:numel(missing_in_G2)
        G2{end+1} = missing_in_G2(i);
    end

    % Number of clusters
    C = numel(G1);
    K = numel(G2);

    % Build contingency table: a(c,k)
    ack = zeros(K, C);
    for k = 1:K
        for c = 1:C
            ack(k, c) = numel(intersect(G2{k}, G1{c}));
        end
    end

    % Compute H(C)
    ac = sum(ack, 1);  % sum over k (rows), gives size of each G1 group
    H_C = 0;
    for c = 1:C
        pc = ac(c) / N;
        if pc > 0
            H_C = H_C - pc * log(pc);
        end
    end

    % Compute H(C | K)
    H_C_given_K = 0;
    for k = 1:K
        a_k = sum(ack(k, :));  % total in G2{k}
        for c = 1:C
            a_ck = ack(k, c);
            if a_ck > 0
                H_C_given_K = H_C_given_K - (a_ck / N) * log(a_ck / a_k);
            end
        end
    end

    % Compute homogeneity
    if H_C == 0
        h = 1;
    else
        h = 1 - H_C_given_K / H_C;
    end
end
