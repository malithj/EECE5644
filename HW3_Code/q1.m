%% EECE5644 - Homework 3 - Question 1
clear all; close all; clc;

mu(1, :) = [-4; 0]; Sigma(:, :, 1) = 0.1 * [10 -4;-4 5];   % mean and covariance of data pdf conditioned on label 1
mu(2, :) = [1; 0]; Sigma(:, :, 2) = 0.1 * [5 2;2 2];       % mean and covariance of data pdf conditioned on label 2
mu(3, :) = [10; 8]; Sigma(:, :, 3) = eye(2);               % mean and covariance of data pdf conditioned on label 3
mu(4, :) = [-5.5; 5.5]; Sigma(:, :, 4) = 0.1 * [4 0;0 4];  % mean and covariance of data pdf conditioned on label 4
rng(50);                                                   % set seed
numC = 6;
priors = [0.50, 0.30, 0.10, 0.10];

%% Gaussian Mixture Model
gm = gmdistribution(mu, Sigma, priors);

sampleSizes = [10, 100, 1000, 10000];
data = containers.Map('KeyType','int32', 'ValueType','any');
labels = containers.Map('KeyType','int32', 'ValueType','any');

for i = 1 : size(sampleSizes, 2)
    [data_arr, labels_arr] = random(gm, sampleSizes(i));
    data(i) = data_arr;
    labels(i) = labels_arr;
end

crossValResults = zeros(size(sampleSizes, 2), numC);

%% EM Algorithm
for i = 1 : size(sampleSizes, 2)
    for c = 1 : numC
        crossValResults(i, c) = crossValidation(data(i), labels(i), c);
    end
end

% results
crossValResults

function error = crossValidation(xTrain, yTrain, numComponents)
    cp = cvpartition(yTrain , 'KFold', 10);
    gm = @(xTrain, yTrain, xTest) predictGM(xTrain, yTrain, xTest, numComponents);
    cvMCR = crossval('mcr', xTrain, yTrain, 'predfun', gm, 'partition', cp);   % misclassification error
    error = cvMCR;
end

function lbls = predictGM(xTrain, yTrain, xTest, numComponents)
     gm = fitgmdist(xTrain, numComponents, 'SharedCovariance', true);
     p = posterior(gm, xTest);
     [~, index] = max(p, [], 2);
     lbls = index;
end