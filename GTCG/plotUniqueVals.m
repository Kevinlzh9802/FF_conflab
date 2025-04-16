function [] = plotUniqueVals(x, fig, normalized, legend)
[C,~,ic] = unique(x);
a_counts = accumarray(ic,1);

figure(fig);
if normalized
    count_normalized = a_counts / length(x);
    plot(C, count_normalized, 'DisplayName',legend);
else
    plot(C, a_counts);
end

end

