function [theta, pos] = getFootOrientation(coords, isLeftHanded)
% GET_FOOT_ORIENTATION Computes foot orientation and average foot position
%
% Inputs:
%   - coords: 1x8 vector [lx, ly, rx, ry, lfx, lfy, rfx, rfy]
%   - isLeftHanded: boolean, true if Y+ is down (left-handed), false if Y+ is up (right-handed)
%
% Outputs:
%   - theta: orientation angle in radians (-pi, pi], with 0 along +X
%   - pos: [x, y] position of the center of all valid points

coords(coords == 0) = NaN;  % treat 0s as missing data

% Extract individual points
lx = coords(1); ly = coords(2);
rx = coords(3); ry = coords(4);
lfx = coords(5); lfy = coords(6);
rfx = coords(7); rfy = coords(8);

left_theta = NaN;
right_theta = NaN;

if all(~isnan([lx, ly, lfx, lfy]))
    dy = lfy - ly;
    dx = lfx - lx;
    left_theta = atan2(dy, dx);
end
if all(~isnan([rx, ry, rfx, rfy]))
    dy = rfy - ry;
    dx = rfx - rx;
    right_theta = atan2(dy, dx);
end

% % Adjust orientation for left-handed system (invert Y direction)
% if isLeftHanded
%     if ~isnan(left_theta)
%         left_theta = atan2(-sin(left_theta), cos(left_theta));
%     end
%     if ~isnan(right_theta)
%         right_theta = atan2(-sin(right_theta), cos(right_theta));
%     end
% end

% Combine orientations
% average direction vector in the correct range
if ~isnan(left_theta) && ~isnan(right_theta)
    theta = atan2(mean([sin(left_theta), sin(right_theta)]), ...
        mean([cos(left_theta), cos(right_theta)]));
elseif ~isnan(left_theta)
    theta = left_theta;
elseif ~isnan(right_theta)
    theta = right_theta;
else
    theta = NaN;
end

% Compute center position of all valid points
all_x = [lx, rx, lfx, rfx];
all_y = [ly, ry, lfy, rfy];
valid = ~isnan(all_x) & ~isnan(all_y);

if any(valid)
    pos = [mean(all_x(valid)), mean(all_y(valid))];
else
    pos = [NaN, NaN];
end
end