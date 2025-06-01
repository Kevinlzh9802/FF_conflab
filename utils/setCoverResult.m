close all;

for i=1:height(data_results)
    head_r = data_results.hipRes{i};
    hip_r = data_results.footRes{i};
    if ~isempty(head_r) & ~isempty(hip_r)
        data_results.cover_idx{i} = setCoverage(head_r, hip_r);
    end
    
end
sum(cell2mat(data_results.cover_idx))