a = figure;
colors = ["red", "blue", "green", "orange"];
clues = ["foot", "hip", "shoulder", "head"];
for k = 1:4
    clue = clues(k);
    color = colors(k);
    plotUniqueVals(results.(clue).group_sizes, a, true, clue, color);
    hold on;
end
hold off
legend

b = figure;
for k=4:7
    subplot(2,2,k-3);
    for clue = ["foot", "hip", "shoulder", "head"]
        gs = results.(clue).group_sizes;
        sp = results.(clue).s_speaker;
        card = gs(gs == k);
        card_ss = sp(gs == k);
        card_ss = card_ss(card_ss <= k);
        card_ss = card_ss(card_ss >= 0);
        % plotUniqueVals(results.(clue).s_speaker, ab);

        [C,~,ic] = unique(card_ss);
        a_counts = accumarray(ic,1);
        normalized = true;
        % figure(fig);
        if normalized
            count_normalized = a_counts / length(card_ss);
            plot(C, count_normalized, 'DisplayName',clue, 'LineWidth',2, ...
                'Marker','o', 'MarkerFaceColor','auto');
            subtitle("Group cardinality " + k);
            xlabel("Simultaneous speakers");
            ylabel("Percentage");
        else
            plot(C, a_counts);
        end
        hold on;
    end
    legend
end
hold off
% legend
c = 0;