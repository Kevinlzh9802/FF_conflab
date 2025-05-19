%%writefile compute_HIC_matrix.m
function HIC = computeHICMatrix(GTgroups, Detgroups)
% COMPUTE_HIC_MATRIX Computes the Histogram of Individuals over Cardinalities (HIC)
%
% Inputs:
%   - GTgroups: cell array where each cell contains a vector of person IDs (ground truth groups)
%   - Detgroups: cell array where each cell contains a vector of person IDs (detected groups)
%
% Output:
%   - HIC: a matrix where HIC(i,j) counts individuals in GT group of size i assigned to Det group of size j

    max_gt = max(cellfun(@length, GTgroups));
    max_det = max(cellfun(@length, Detgroups));
    HIC = zeros(max_gt, max_det);

    % Flatten person-group associations
    person_to_gt = containers.Map('KeyType','int32','ValueType','int32');
    for i = 1:length(GTgroups)
        for p = GTgroups{i}
            person_to_gt(p) = length(GTgroups{i});
        end
    end

    for j = 1:length(Detgroups)
        group = Detgroups{j};
        det_card = length(group);
        for p = group
            if isKey(person_to_gt, p)
                gt_card = person_to_gt(p);
                HIC(gt_card, det_card) = HIC(gt_card, det_card) + 1;
            end
        end
    end

    % Normalize HIC by rows
    for i = 1:size(HIC,1)
        if sum(HIC(i,:)) > 0
            HIC(i,:) = HIC(i,:) / sum(HIC(i,:));
        end
    end
end

