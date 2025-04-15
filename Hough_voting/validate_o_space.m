function is_valid = validate_o_space(o_center, all_samples, labels, ...
    members, weights, r, tau_intr)
% Check for intruders inside o-space (circle of radius r)
distances = sqrt((all_samples(:,1) - o_center(1)).^2 + (all_samples(:,2) - o_center(2)).^2);

closeness = distances < r;
intruder = ~ismember(labels, members);
intrude_weight = weights > tau_intr;

is_valid = ~any(intruder & closeness & intrude_weight);
% for i = 1:length(distances)
%     if distances(i) < r && ~ismember(labels(i), members) && weights(i) > tau_intr
%         is_valid = false;
%         return;
%     end
% end
% is_valid = true;
end
