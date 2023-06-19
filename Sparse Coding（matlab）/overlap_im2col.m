function X = overlap_im2col(I, bb, overlap)
% Extracts (bb x bb) patches from image I after every 'overlap' pixels
%
% I: (N x M) image
% bb: patch size
% overlap: number of pixels between consecutive patches
%

% Get image size
[N, M] = size(I);
% Initialise X  初始化x
X = [];
% Calculate how many patches fit horisontally and vertically  计算出有多少补丁水平和垂直方向
row = (N - bb)/overlap;
col = (M - bb)/overlap;

% Iterate through patches  迭代通过补丁
for i = 0:row
    for j = 0:col
        % Take a patch with top-left corner at (i, j) and vectorise it
        % 采取与左上角补丁位于（i，j）和vectorise..
        % fprintf('%bb  %bb | %bb  %bb \n',1+i*overlap,bb+i*overlap,1+j*overlap,bb+j*overlap)
        X = [X, reshape(I([1:bb]+i*overlap, [1:bb]+j*overlap), bb*bb, 1)];
    end
end

