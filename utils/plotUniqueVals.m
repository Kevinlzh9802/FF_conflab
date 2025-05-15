function [] = plotUniqueVals(x, fig, normalized, legend, color)
    [C,~,ic] = unique(x);
    a_counts = accumarray(ic,1);
    
    figure(fig);
    if normalized
        count_normalized = a_counts / length(x);
        plot(C, count_normalized, 'DisplayName',legend, 'LineWidth',2, ...
            'Marker','o', 'MarkerFaceColor','auto');
        % subtitle("Group cardinality");
        xlabel("Detected group size");
        ylabel("Percentage");
    else
        plot(C, a_counts);
    end
    
    end
    