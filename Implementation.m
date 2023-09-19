clear;close all;clear all, clc;
% Generate data matrix using gen_clusterdata function ID = 10749174;
X = gen_clusterdata(ID);
% Get number of rows (objects or cases) in data matrix
N = size(X,1);
fprintf('\nTotal Number of Rows = %f\n\n',N);
% Loop through each column (feature) in data matrix
for i = 1:size(X,2)
    % Get mean and standard deviation of column
    mu = mean(X(:,i));
    stdDev = std(X(:,i));
    % Print mean and standard deviation
fprintf('Column %d: Mean = %f, Standard deviation = %f\n', i, mu, stdDev);
    % Plot histogram of column
    figure
    histogram(X(:,i))
     % Add labels and title
xlabel('Data');
ylabel('Frequency');
title(sprintf('Column %d: Mean = %.2f, Std = %.2f', i, mu, stdDev));
end
% Get covariance matrix of data matrix
C = cov(X);
% Print covariance matrix
fprintf('\nCovariance matrix:\n') disp(C)
% Get correlation matrix of data matrix
R = corr(X);
% Print correlation matrix
fprintf('Correlation matrix:\n') disp(R)
% Set range of values for K
K_range = 3:5;
% Preallocate variable to store silhouette scores
s_all = zeros(length(K_range), 1);
% Loop over values of K
for i = 1:length(K_range)
    % Perform K-means clustering
    [IDX, C] = kmeans(X, K_range(i));
    % Compute silhouette score
    s = silhouette(X, IDX);
    % Store silhouette score
s_all(i) = mean(s);
% Plot silhouette scores for each cluster
figure
silhouette(X, IDX)
title(sprintf('Silhouette scores for K = %d', K_range(i)))
% Plot clusters and cluster centroids
figure
gscatter(X(:,1), X(:,2), IDX)
hold on
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3) if K_range(i) < 3
legend(sprintf('K = %d', K_range(i)), 'Location','best')
elseif K_range(i) == 3
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'best') elseif K_range(i) == 4
    legend('Cluster
'best')
else
    legend('Cluster
'Location', 'best')
end
1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Centroids', 'Location', 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids',
title(sprintf('K-means clustering for K = %d', K_range(i))) end
% Find index of K with highest silhouette score
[~, idx] = max(s_all);
% Print mean Silhouette scores for each K value
fprintf('Mean silhouette scores:\n') disp(s_all)
% Print optimal number of clusters
fprintf('Optimal number of clusters: %d\n', K_range(idx))