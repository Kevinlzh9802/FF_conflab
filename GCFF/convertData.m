clear variables; close all;
file_name = "../data/head.mat";
load(file_name, 'all_data');
save('mydata.mat','all_data','-v7.3'); 
% fid = fopen('data.jsonl','w');
% for i = 1:height(all_data)
%     row = table2struct(all_data(i,:));
%     % ensure arrays are regular MATLAB arrays, not tables
%     row.Features = all_data.Features{i};      % 1x48 double -> JSON list
%     row.GT = all_data.GT{i};                  % 1x3 cell    -> JSON list
%     fprintf(fid, '%s\n', jsonencode(row));
% end
% fclose(fid);