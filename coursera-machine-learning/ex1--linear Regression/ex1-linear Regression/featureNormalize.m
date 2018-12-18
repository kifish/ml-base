function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));  %%1 row, (size(X,2)=the columns of X) columns
sigma = zeros(1, size(X, 2));  

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
for iter = 1:size(X, 2)
    mu(1, iter) = mean(X(:,iter));
    sigma(1, iter) = std(X(:,iter));
    X_norm(:, iter) = (X_norm(:, iter) - mu(1, iter)) ./ sigma(1, iter);
end;

%%maybe the following is faster

%mu = mean(X);
%sigma = std(X);
%for i=1:size(X,1)
%   X_norm(i, :) = (X(i, :) - mu) ./ sigma;
%end
%%

%%trade-off  (a litte confused)
%mu = mean(X);
%sigma = std(X);
%mu2 = repmat(mu, size(X,1), 1);    mu2: size(X,1) rows, 1 columns
%sigma2 = repmat(sigma, size(X,1), 1);

%X_norm = (X - mu2) ./ sigma2;
%%

% ============================================================

end
