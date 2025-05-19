%%writefile grode_metrics.m
function metrics = grodeMetrics(HIC)
% GRODE_METRICS Computes GRODE metrics from the HIC matrix
%
% Input:
%   - HIC: normalized HIC matrix (rows sum to 1)
%
% Output:
%   - metrics: struct with fields A, Pr, Re, F1, D, UL, WUL

    n = size(HIC, 1);
    diag_vals = diag(HIC);

    % Accuracy A
    A = sum(diag_vals);

    % Precision, Recall, F1 per cardinality
    Pr = zeros(n, 1);
    Re = zeros(n, 1);
    F1 = zeros(n, 1);
    for c = 1:n
        denom_pr = sum(HIC(:,c));
        denom_re = sum(HIC(c,:));
        Pr(c) = HIC(c,c) / denom_pr * (denom_pr > 0);
        Re(c) = HIC(c,c) / denom_re * (denom_re > 0);
        if Pr(c)+Re(c) > 0
            F1(c) = 2 * Pr(c) * Re(c) / (Pr(c) + Re(c));
        else
            F1(c) = 0;
        end
    end

    % Cardinality deviation D
    mu = mean(diag_vals);
    D = sqrt(mean((diag_vals - mu).^2));

    % UL and WUL
    UL = 0;
    WUL = 0;
    for i = 1:n
        for j = 1:n
            if j > i
                UL = UL + HIC(i,j);
                WUL = WUL + HIC(i,j) * (j-i);
            elseif j < i
                UL = UL - HIC(i,j);
                WUL = WUL - HIC(i,j) * (i-j);
            end
        end
    end

    metrics = struct('A', A, 'Pr', Pr, 'Re', Re, 'F1', F1, ...
                     'D', D, 'UL', UL, 'WUL', WUL);
end


