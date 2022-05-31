function [theta] = FixRangeOfAngles(theta)
% Input: degrees
% Output: radians in -180 to 180
theta = atan2(sind(theta),cosd(theta));
end