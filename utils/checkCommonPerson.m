close all;
views = cell(4, 1);
for k=1:4
    kc = 2*k;
    views{k} = filterTable(all_data, [kc], [3], [1]);
end

[hmax,mc] = max([height(views{1}), height(views{2}), height(views{3}), height(views{4})]);

a1 = match_video_frames(views{1}, [15,7,11,0.834168], views{2}, [15,7,13,0.967634], 59.94);
a2 = match_video_frames(views{2}, [15,7,13,0.967634], views{3}, [15,7,11,0.834168], 59.94);


for i=1:hmax
    ts = views{mc}.Timestamp(i);
    if i>2
        ts2 = views{mc}.Timestamp(i-2);
    else
        ts2 = 0;
    end
    v1 = views{1}(views{1}.Timestamp == ts, :);
    v2 = views{2}(views{2}.Timestamp == ts2, :);
    v3 = views{3}(views{3}.Timestamp == ts, :);
    v4 = views{4}(views{4}.Timestamp == ts, :);
    
    g2 = [];
    g4 = [];
    g6 = [];
    g8 = [];

    if ~isempty(v1)
        g2 = v1.Features{1}(:,1);
    end
    if ~isempty(v2)
        g4 = v2.Features{1}(:,1);
    end
    if ~isempty(v3)
        g6 = v3.Features{1}(:,1);
    end
    if ~isempty(v4)
        g8 = v4.Features{1}(:,1);
    end
    c1 = intersect(g2, g4);
    c2 = intersect(g4, g6);
    c3 = intersect(g6, g8);
    if ~isempty(c1)
        c1
    end
    if ~isempty(c2) && v2.Vid(1) == v3.Vid(1)
        c2
        % v2.Timestamp(1)
        v2.Timestamp
        v3.Timestamp
        f2 = v2.Features{1}(:,21:40);
        f3 = v3.Features{1}(:,21:40);
        ind2 = (f2(:,1)==c2);
        ind3 = (f3(:,1)==c2);

        % xy2 = reshape(f2(ind2, 5:end), [2,8])';
        % xy3 = reshape(f3(ind3, 5:end), [2,8])';
        % figure;
        % scatter(xy2(:,1), xy2(:,2))
        % hold on;
        % scatter(xy3(:,1), xy3(:,2))
        % c = 9;
        hfig = figure;
        plotSkeletonOnImage(axes(hfig), 0, f2(ind2, [1, 5:end]), [1,2,3,4], true);
        hold on;
        plotSkeletonOnImage(axes(hfig), 0, f3(ind3, [1, 5:end]), [1,2,3,4], true);
    end
    if ~isempty(c3)
        c3
    end
end
