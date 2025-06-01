M = cell2mat(max_speaker{7});

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
x = 1:size(M,1);
figure;
bar(x, row_means, 'FaceColor', [0.2 0.6 0.8]);
hold on;
errorbar(x, row_means, row_stds, 'k.', 'LineWidth', 1.5);
xlabel('Row index');
ylabel('Expected Value ± Std Dev');
title('Distribution-based Mean and Std per Row');
grid on;