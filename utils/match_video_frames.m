%%writefile match_video_frames.m
function matches = match_video_frames(table1, timecode1, table2, timecode2, fps)
% MATCH_VIDEO_FRAMES Matches frames between two videos with a given latency
%
% Inputs:
%   - table1: MATLAB table with columns 'Features' and 'Timestamp' (in frames) for video 1
%   - timecode1: 1x4 vector [HH MM SS MS] for video 1 start
%   - table2: same as table1, for video 2
%   - timecode2: 1x4 vector [HH MM SS MS] for video 2 start
%   - fps: frame rate (e.g., 60)
%
% Output:
%   - matches: Nx2 array where each row is [index1, index2] for matched frames

    % Convert timecodes to seconds
    start1_sec = timecode1(1)*3600 + timecode1(2)*60 + timecode1(3) + timecode1(4)/100;
    start2_sec = timecode2(1)*3600 + timecode2(2)*60 + timecode2(3) + timecode2(4)/100;

    % Convert frame timestamps to absolute times (in seconds)
    time1 = start1_sec + double(table1.Timestamp) / fps;
    time2 = start2_sec + double(table2.Timestamp) / fps;

    matches = [];

    for i = 1:length(time1)
        t1 = time1(i);
        [delta, j] = min(abs(time2 - t1));

        if delta <= 1 / fps
            matches(end+1, :) = [i, j]; %#ok<AGROW>
        end
    end
end