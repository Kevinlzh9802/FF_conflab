clear variables; close all;
data_path = "../data/";
speaking_path = data_path + "processed/speaking/";
confidence_path = data_path + "processed/confidence/";

speaking_status = struct();
speaking_status.speaking = read_csvs(speaking_path);
speaking_status.confidence = read_csvs(confidence_path);
save("../data/speaking_status.mat", 'speaking_status');

function result = read_csvs(path)

Files=dir(path + "*.csv"); % edit your own path to the pose data!!!
for k=1:length(Files)
    FileName=Files(k).name;
    data = readmatrix(strcat(path, FileName));
    fieldName = FileName(1:9);
    result.(fieldName) = data;
end

end