clear variables; close all;
parent_dir = '../data/export'; mkdir(parent_dir); 

for clue = ["head", "shoulder", "hip", "foot"]
    file_name = sprintf("../data/%s.mat", clue);
    load(file_name, 'all_data');
    write_feature_data(all_data, clue, parent_dir);
end

function write_feature_data(T, clue, parent_dir)
    outdir = fullfile(parent_dir, clue);
    mkdir(outdir);
    mkdir(fullfile(outdir, 'features'));
    
    N = height(T);
    row_id = (1:N).';
    metadata = table(row_id, T.Cam, T.Vid, T.Seg, T.Timestamp, ...
        'VariableNames', {'row_id','Cam','Vid','Seg','Timestamp'});
    writetable(metadata, fullfile(outdir,'metadata.csv'));
    
    % features: write each n×48 matrix to its own CSV
    for i = 1:N
        Fi = T.Features{i};                 % numeric [n×48], n may vary
        writematrix(Fi, fullfile(outdir,'features', sprintf('f_%06d.csv', i)));
    end
    
    % GT: JSON Lines (list of lists)
    fid = fopen(fullfile(outdir,'gt.jsonl'),'w');
    for i = 1:N
        GTi = T.GT{i};                      % k×1 cell, each cell numeric row vector
        GT_lists = cellfun(@(v) reshape(v,1,[]), GTi, 'UniformOutput', false);
        obj = struct('row_id', i, 'GT', {GT_lists});
        fprintf(fid, '%s\n', jsonencode(obj));
    end
    fclose(fid);
end

