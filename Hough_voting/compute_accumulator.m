function Atilde = compute_accumulator(AI, AL)
% Adjust vote strength by number of unique voters
Atilde = zeros(size(AI));
for i = 1:size(AI,1)
    for j = 1:size(AI,2)
        if ~isempty(AL{i,j})
            Atilde(i,j) = AI(i,j) * length(AL{i,j});
        end
    end
end
end
