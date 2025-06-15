close all;

hm = zeros(4);
for k1=1:4
    for k2=k1+1:4
        g1 = 0;
        g2 = 0;
        for i=1:height(data_results)
            r1 = clues(k1) + "Res";
            r2 = clues(k2) + "Res";
            s1 = data_results.(r1){i};
            s2 = data_results.(r2){i};
            if ~isempty(s1) & ~isempty(s2)
                data_results.cover_idx{i} = setCoverage(s1, s2);
                g1 = g1 + length(s1);
                g2 = g2 + length(s2);
            end
            
        end
        % r1,r2
        scores = sum(cell2mat(data_results.cover_idx)) ./ [g1, g2];
        hm(k1, k2) = scores(1);
        hm(k2, k1) = scores(2);
    end
end