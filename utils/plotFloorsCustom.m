function plotFloorsCustom(max_speaker, base_clue, outdir)
% Plot floor analysis results for a given base_clue
% Inputs:
%   max_speaker - cell array containing speaker count data
%   base_clue - string indicating the clue type
%   outdir - output directory path

% Use the current figure (created in detectSubFloor)
% Don't close all or create new figure
for k=1:4
sp_num = k+3;

M = cell2mat(max_speaker{sp_num});

% Define bin values: assuming M(:,1) is count of 0s, M(:,2) = count of 1s, etc.
bins = 0:(size(M,2)-1);

% Preallocate mean and std vectors
row_means = zeros(size(M,1),1);
row_stds  = zeros(size(M,1),1);

% Compute mean and std treating each row as a distribution
for i = 1:size(M,1)
    counts = M(i, :);
    total = sum(counts);
    if total > 0
        probs = counts / total;
        mu = sum(probs .* bins);
        sigma = sqrt(sum(probs .* (bins - mu).^2));
    else
        mu = NaN;
        sigma = NaN;
    end
    row_means(i) = mu;
    row_stds(i) = sigma;
end

% Plotting
subplot(1, 4, k);
x = 1:size(M,1);

bar(x, row_means, 'FaceColor', [0.2 0.6 0.8]);
hold on;
errorbar(x, row_means, row_stds, 'k.', 'LineWidth', 1.5);
xlabel('Duration (s)');
ylabel('Simultaneous speaker');
title("Group Cardinality" + num2str(sp_num));
grid on;

end

% Save the figure
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
saveas(gcf, fullfile(outdir, sprintf('floor_analysis_%s.png', base_clue)));
fprintf('Floor analysis plot saved for %s\n', base_clue);