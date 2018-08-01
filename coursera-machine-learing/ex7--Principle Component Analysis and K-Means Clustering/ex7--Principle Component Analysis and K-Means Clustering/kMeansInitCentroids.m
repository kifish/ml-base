function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

randidx = randperm(size(X,1)); %Matlab自带函数randperm(n)产生1到n的整数的无重复的随机排列，利用它就可以得到无重复的随机数。
centroids = X(randidx(1:K),:); %select randomly K rows of X as centroids






% =============================================================

end

