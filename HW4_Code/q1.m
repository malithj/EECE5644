%% EECE5644 - Homework 4 - Question 1
clear all; close all; clc;
colorPlane = imread('3096_colorPlane.jpg');
colorBird = imread('42049_colorBird.jpg');
height = size(colorPlane, 1);
width = size(colorPlane, 2);
indicesX = repmat([1 : height], width, 1).';
indicesY = repmat([1 : width], height, 1);

%% rescale to values between 0 and 1
rescaledColorPlane = rescale(colorPlane, 0, 1);
rescaledColorBird = rescale(colorBird, 0, 1);
rescaledIndicesX = rescale(indicesX, 0, 1);
rescaledIndicesY = rescale(indicesY, 0, 1);

%% create the feature vector by concatenation
%% height x width x dim 
featuresColorPlane = cat(3, cat(3, rescaledColorPlane, rescaledIndicesX), rescaledIndicesY);
featuresColorPlane = reshape(reshape(featuresColorPlane,1,[])', height * width, []); % reshape to extract feature vectors and remove 2D information
featuresColorBird = cat(3, cat(3, rescaledColorBird, rescaledIndicesX), rescaledIndicesY);
featuresColorBird = reshape(reshape(featuresColorBird,1,[])', height * width, []); % reshape to extract feature vectors and remove 2D information

%% kmeans clustering
for k = 2:5
    [idxPlane, ~] = kmeans(featuresColorPlane, k);
    [idxBird, ~] = kmeans(featuresColorBird, k);
    
    %% plot results
    %% column major transformation
    figure
    labels = reshape(idxPlane, height, width);
    imagesc(labels);
    colormap(hsv(k));
    colorbar('Ticks',1:k);
    title(['Plane - KMeans Clustering (Num Clusters: ', num2str(k), ')']);
    xlabel('Width','FontSize',14);
    ylabel('Height','FontSize',14);
    filename = sprintf('plane_num_clusters_%d.jpg', k);
    saveas(gcf, filename);
    
    figure
    labels = reshape(idxBird, height, width);
    imagesc(labels);
    colormap(hsv(k));
    colorbar('Ticks',1:k);
    title(['Bird - KMeans Clustering (Num Clusters: ', num2str(k), ')']);
    xlabel('Width','FontSize',14);
    ylabel('Height','FontSize',14);
    filename = sprintf('bird_num_clusters_%d.jpg', k);
    saveas(gcf, filename);
end

%% Gaussian Mixture Model
for k = 2:5
    gm = fitgmdist(featuresColorPlane, k);
    p = posterior(gm, featuresColorPlane);
    [~, idxPlane] = max(p, [], 2);
    gm = fitgmdist(featuresColorBird, k);
    p = posterior(gm, featuresColorBird);
    [~, idxBird] = max(p, [], 2);
    
    
    %% plot results
    figure
    lbls = reshape(idxPlane, height, width);
    imagesc(lbls);
    colormap(hsv(k));
    colorbar('Ticks',1:k);
    title(['Plane - GMM Clustering (Num Clusters: ', num2str(k), ')']);
    xlabel('Width','FontSize',14);
    ylabel('Height','FontSize',14);
    filename = sprintf('gmm_plane_num_clusters_%d.jpg', k);
    saveas(gcf, filename);
    
    figure
    lbls = reshape(idxBird, height, width);
    imagesc(lbls);
    colormap(hsv(k));
    colorbar('Ticks',1:k);
    title(['Bird - GMM Clustering (Num Clusters: ', num2str(k), ')']);
    xlabel('Width','FontSize',14);
    ylabel('Height','FontSize',14);
    filename = sprintf('gmm_bird_num_clusters_%d.jpg', k);
    saveas(gcf, filename);
end

