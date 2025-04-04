function [AI, AL, sz] = vote_for_o_space(samples, weights, labels, r, grid_size)
% Each sample votes for an o-space center based on direction
x_votes = samples(:,1) + r * cos(samples(:,3));
y_votes = samples(:,2) + r * sin(samples(:,3));

x_idx = round(x_votes / grid_size);
y_idx = round(y_votes / grid_size);

max_x = max(x_idx) + 1;
max_y = max(y_idx) + 1;

AI = zeros(max_y, max_x);
AL = cell(max_y, max_x);
sz = [max_y, max_x];

for i = 1:length(weights)
    xi = x_idx(i);
    yi = y_idx(i);
    AI(yi, xi) = AI(yi, xi) + weights(i);
    if isempty(AL{yi, xi})
        AL{yi, xi} = labels(i);
    elseif ~ismember(labels(i), AL{yi, xi})
        AL{yi, xi}(end+1) = labels(i);
    end
end
end
