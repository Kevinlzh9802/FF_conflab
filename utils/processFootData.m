function T = processFootData(T, isLeftHanded)
% PROCESS_FOOT_DATA Computes foot positions and orientations for each frame
%
% Inputs:
%   - T: table with column 'footFeature', each row is an n x 48 matrix
%   - isLeftHanded: boolean, true if Y+ is down (left-handed)
%
% Output:
%   - T: modified table with updated footFeature values in-place

    numRows = height(T);

    for i = 1:numRows
        features = T.footFeat{i};
        if isempty(features)
            continue;
        end

        % Split into two n x 24 matrices
        M1 = features(:, 1:24);
        M2 = features(:, 25:48);

        for j = 1:size(M1,1)
            [theta1, pos1] = getFootOrientation(M1(j,17:24), isLeftHanded);
            M1(j,2:3) = pos1 .* [1920, 1080];
            M1(j,4) = theta1;

            [theta2, pos2] = getFootOrientation(M2(j,17:24), isLeftHanded);
            M2(j,2:3) = pos2;
            M2(j,4) = theta2;
        end

        % Recombine and update the table
        T.footFeat{i} = [M1, M2];
    end
end