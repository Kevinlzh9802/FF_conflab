function Atilde = compute_accumulator(AI, AL)
% Adjust vote strength by number of unique voters
Atilde = zeros(size(AI));
if length(size(AL)) == 2
    Atilde = AI .* AL;
elseif length(size(AL)) == 3
    Atilde = AI .* sum(AL, 3);
end
% for i = 1:size(AI,1)
%     for j = 1:size(AI,2)
%         if ~isempty(AL{i,j})
%             Atilde(i,j) = AI(i,j) * length(AL{i,j});
%         end
%     end
% end
end
