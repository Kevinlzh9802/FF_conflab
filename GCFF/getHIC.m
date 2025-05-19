for k=1:height(all_data)
    all_data.HIC(k) = computeHICMatrix(all_data.GT(k), all_data.hipRes(k));
end