% Set publication figure style (equivalent to set_pubfig)
function set_pubfig()
    set(groot, 'DefaultAxesFontSize', 22);
    set(groot, 'DefaultAxesTitleFontWeight', 'normal');
    set(groot, 'DefaultAxesTitleFontSizeMultiplier', 1);
    set(groot, 'DefaultAxesLabelFontSizeMultiplier', 1);
    set(groot, 'DefaultLineLineWidth', 2);
    set(groot, 'DefaultAxesFontName', 'Times New Roman');
end

% Load and process max windowed floor data
function df = max_windowed_floors(path)
    load(path, 'df'); % assuming .mat file contains a table named df

    non_id_cols = ~ismember(df.Properties.VariableNames, {'cardinality', 'id'});
    df.max_floors = max(df{:, non_id_cols}, [], 2);

    [~, name, ~] = fileparts(path);
    df.window_size = repmat(str2double(name), height(df), 1);
    
    df = df(:, {'id', 'cardinality', 'window_size', 'max_floors'});
end

% Plot floors as line plots grouped by cardinality
function plot_floors(df)
    df.window_size = df.window_size / 20;
    cardinalities = unique(df.cardinality);

    figure;
    t = tiledlayout(1, length(cardinalities), 'TileSpacing', 'compact');
    title(t, 'Variation in Number of Conversation Floors with Speaking Duration Threshold');

    for i = 1:length(cardinalities)
        nexttile;
        card = cardinalities(i);
        subdf = df(df.cardinality == card, :);

        means = grpstats(subdf.max_floors, subdf.window_size, 'mean');
        stds = grpstats(subdf.max_floors, subdf.window_size, 'std');
        win_sizes = unique(subdf.window_size);

        errorbar(win_sizes, means, stds, '-o');
        title(sprintf('Cardinality %d', card));
        xlabel('Speaking Duration in seconds');
        ylabel('Mean (Max. No. of Distinct Floors)');
        xlim([0, max(win_sizes)]);
        xticks(1:1:20);
    end
end

% Plot number of F-formation samples per window size and cardinality
function plot_cardinality_counts(df)
    df.window_size = df.window_size / 20;

    figure;
    grouped_counts = groupsummary(df, {'window_size', 'cardinality'}, 'numel');

    unq_windows = unique(grouped_counts.window_size);
    unq_cards = unique(grouped_counts.cardinality);
    count_matrix = zeros(length(unq_windows), length(unq_cards));

    for i = 1:height(grouped_counts)
        row = find(unq_windows == grouped_counts.window_size(i));
        col = find(unq_cards == grouped_counts.cardinality(i));
        count_matrix(row, col) = grouped_counts.GroupCount(i);
    end

    bar(unq_windows, count_matrix);
    xlabel('Speaking Duration in seconds');
    ylabel('Number of F-formations');
    xticks(1:1:20);
    yticks(0:1:22);
    legend(arrayfun(@(x) sprintf('Cardinality %d', x), unq_cards, 'UniformOutput', false));
    title('Number of F-formation Samples at Varying Turn Durations');
end