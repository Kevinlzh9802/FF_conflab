close all;
gt_sizes = get_group_sizes(data_results.GT);
gt_sizes = gt_sizes(gt_sizes > 1);

% a = figure;
% colors = ["red", "blue", "green", "orange"];
% clues = ["foot", "hip", "shoulder", "head"];
% for k = 1:4
%     clue = clues(k);
%     color = colors(k);
%     plotUniqueVals(results.(clue).group_sizes, a, true, clue, color);
%     hold on;
% end
% 
% plotUniqueVals(gt_sizes, a, true, "GT", "black");
% hold off
% legend

plotPercDiffs(results, gt_sizes);

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

function g_sizes = get_group_sizes(groups)
g_sizes = [];
for k=1:length(groups)

    g_numbers = length(groups{k});
    for j=1:g_numbers
        g_sizes = [g_sizes, length(groups{k}{j})];
    end
end

end

function diffs = plotPercDiffs(results, gt_sizes)
colors = ["red", "blue", "green", "orange"];
clues = ["foot", "hip", "shoulder", "head"];
sizes_num = length(unique(gt_sizes));
diffs = zeros(5, sizes_num);

[unique_gt, ~, ic] = unique(gt_sizes);
a_counts = accumarray(ic,1);
diffs(5, :) = pad_with_zeros(a_counts / length(gt_sizes), sizes_num);
for k = 1:4
    clue = clues(k);
    color = colors(k);

    x = results.(clue).group_sizes;
    [unique_x, ~, ic] = unique(x);
    a_counts = accumarray(ic,1);
    count_normalized = (a_counts / length(x))';
    diffs(k, :) = pad_with_zeros(count_normalized, sizes_num) - diffs(5, :);
    c = 9;
end
bar((diffs(1:end-1,:))');        % Transpose to group by column
xlabel('Group (Columns)')
group_labels = {'1', '2', '3', '4', '5', '6', '7', '8'};
ylabel('Value')
legend(clues)
xticks(1:size(diffs,2));
xticklabels(group_labels)
title('Grouped Bar Chart by Column')

end
function y = pad_with_zeros(x, N)
    if isrow(x)
        xt = x';  % convert back to row if needed
    else
        xt = x;
    end
    % Pads vector x with zeros on the right to length N
    if length(xt) < N
        y = [xt; zeros(N - length(xt), 1)];  % column vector
        
    else
        y = xt(1:N);  % optionally trim if x is longer
    end
    if isrow(x)
        y = y';  % convert back to row if needed
    end
end