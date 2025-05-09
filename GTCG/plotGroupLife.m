
% clear variables; close all;
addpath(genpath('../utils'));
all_data = load("comb_from_foot.mat");
all_data = all_data.a;
used_data = filterTable(all_data, [4], [3], [2]);

plotGroupSizes(used_data);

function a = plotGroupSizes(T)
colNames = T.Properties.VariableNames;
colsToAnalyze = {'GT', 'footRes', 'hipRes', 'shoulderRes', 'headRes'};
numRows = height(T);
numCols = numel(colsToAnalyze);
% Preallocate matrix for lengths
lengths = zeros(height(T), numCols);

for colIdx = 1:numCols
    colName = colsToAnalyze{colIdx};
    for row = 1:height(T)
        cellContent = T{row, colName};
        if iscell(cellContent) || isnumeric(cellContent)
            lengths(row, colIdx) = numel(cellContent{1});
        else
            lengths(row, colIdx) = NaN;
        end
    end
end

% Plotting
figure;
plot(1:numRows, lengths, '-', 'LineWidth', 1.5);
xlabel('Row index');
ylabel('Number of elements in each cell');
legend(colsToAnalyze, 'Interpreter', 'none', 'Location', 'best');
title('Length of Each Cell Element per Table Column');
end
