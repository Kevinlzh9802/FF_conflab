function [AI, AL, sz] = vote_for_o_space(samples, weights, labels, ...
    num_person, r, grid_size)
% Each sample votes for an o-space center based on direction
x_votes = samples(:,1) + r * cos(samples(:,3));
y_votes = samples(:,2) + r * sin(samples(:,3));

x_idx = round(x_votes);
y_idx = round(y_votes);

max_x = grid_size(1);
max_y = grid_size(2);

AI = zeros(max_y, max_x);
AL = zeros(max_y, max_x, num_person);
sz = [max_y, max_x];

for i = 1:length(weights)
    xi = x_idx(i);
    yi = y_idx(i);
    if (xi > 0) && (xi <= max_x) && (yi > 0) && (yi <= max_y)
        AI(yi, xi) = AI(yi, xi) + weights(i);
        AL(yi, xi, labels(i)) = 1;
        % if isempty(AL{yi, xi})
        %     AL{yi, xi} = labels(i);
        % elseif ~ismember(labels(i), AL{yi, xi})
        %     AL{yi, xi}(end+1) = labels(i);
        % end
    end
end
end
