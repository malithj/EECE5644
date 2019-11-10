%% EECE5644 - Homework 4 - Question 2
clear all; close all; clc;
mu = [0 0];
sigma = eye(2);
rng('default');  % set the seed
N = 1000;
%% Points from Gaussian Distribution
classData(:, :, 1) = mvnrnd(mu, sigma, N);

%% Points Randomly Distributed Within Circle
rMin = 2; rMax = 3;
radius = (rMax - rMin) .* rand(N, 1) + rMin;
angle = (pi - (-pi)) .* rand(N, 1) - pi;
[x y] = pol2cart(angle, radius);
classData(:, :, 2) = cat(2, x, y);

classPriors = [0.35, 0.65]; thr = [0, cumsum(classPriors)];
u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1), clf, colorList = 'brg';

%% Selection based on Priors
for l = 1:2
    indices = find(thr(l) <= u & u < thr(l + 1)); % u precisely 1 not handled
    if l == 2 && sum(1 == u > 0)                  % u precisely 1 handled by classifying as the last label
        indices_u_1 = find(1 == u);
        indices = [indices, indices_u_1];
    end
    L(1, indices) = l * ones(1, length(indices));
    x(:, indices) = classData(indices, :, l).';
    figure(1), plot(x(1, indices), x(2, indices), '.', 'MarkerFaceColor', colorList(l)); axis equal, hold on,
end
title('Data Distribution');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'Gaussian Distribution', 'Random Vectors in Circular Region'});
filename = sprintf('data_distribution.jpg');
saveas(gcf, filename);

cp = cvpartition(L , 'KFold', 10);
mdlLin = fitcsvm(x', L,'Standardize', true, 'ClassNames', [1 2]);
valLin = crossval(mdlLin, 'CVPartition', cp);   % validation
mdlGau = fitcsvm(x', L,'Standardize', true, 'ClassNames', [1 2], 'KernelFunction','gaussian');
valGau = crossval(mdlGau, 'CVPartition', cp);   % validation

kfoldLoss(valGau)
kfoldLoss(valLin)
rMdlLin = fitcsvm(x', L, 'ClassNames', [1 2], 'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
rMdlGau = fitcsvm(x', L, 'ClassNames', [1 2], 'KernelFunction', 'gaussian', 'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
valLin = crossval(mdlLin, 'CVPartition', cp);   % validation
valGau = crossval(mdlGau, 'CVPartition', cp);   % validation

%% Visualize the Classifier
[predictedL, ~] = predict(rMdlLin, x');
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(x(1, :)) : d:max(x(1, :)),...
    min(x(2, :)) : d:max(x(2, :)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(rMdlLin, xGrid);
figure;
h = nan(3, 1); 
h(1:2) = gscatter(x(1, :).', x(2, :).', predictedL, 'rg', '+*');
hold on
h(3) = plot(x(1, rMdlLin.IsSupportVector).', x(2, rMdlLin.IsSupportVector).', 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');
legend(h, {'Predicted Gaussian','Predicted Random Vectors in Circular Region', 'Support Vectors'}, 'Location', 'southoutside');
axis equal
hold off
filename = sprintf('linear_svm_classification.jpg');
saveas(gcf, filename);
misclassification_rate = kfoldLoss(valLin)

%% Visualize the Classifier
[predictedL, ~] = predict(rMdlGau, x');
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(x(1, :)) : d:max(x(1, :)),...
    min(x(2, :)) : d:max(x(2, :)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(rMdlGau, xGrid);
figure;
h = nan(3, 1);
h(1:2) = gscatter(x(1, :).', x(2, :).', predictedL, 'rg', '+*');
hold on
h(3) = plot(x(1, rMdlGau.IsSupportVector).', x(2, rMdlGau.IsSupportVector).', 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');
legend(h, {'Predicted Gaussian','Predicted Random Vectors in Circular Region', 'Support Vectors'}, 'Location', 'southoutside');
axis equal
hold off
filename = sprintf('gaussian_svm_classification.jpg');
saveas(gcf, filename);
misclassification_rate = kfoldLoss(valGau)

%% Generate Test Dataset
%% Points from Gaussian Distribution
rng(101); % change seed to generate a different data set
classData(:, :, 1) = mvnrnd(mu, sigma, N);

%% Points Randomly Distributed Within Circle
rMin = 2; rMax = 3;
radius = (rMax - rMin) .* rand(N, 1) + rMin;
angle = (pi - (-pi)) .* rand(N, 1) - pi;
[x y] = pol2cart(angle, radius);
classData(:, :, 2) = cat(2, x, y);
u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure, clf, colorList = 'brg';

%% Selection based on Priors
for l = 1:2
    indices = find(thr(l) <= u & u < thr(l + 1)); % u precisely 1 not handled
    if l == 2 && sum(1 == u > 0)                  % u precisely 1 handled by classifying as the last label
        indices_u_1 = find(1 == u);
        indices = [indices, indices_u_1];
    end
    L(1, indices) = l * ones(1, length(indices));
    x(:, indices) = classData(indices, :, l).';
    plot(x(1, indices), x(2, indices), '.', 'MarkerFaceColor', colorList(l)); axis equal, hold on,
end
title('Test Data Distribution');
xlabel('Feature 1');
ylabel('Feature 2');
legend({'Gaussian Distribution', 'Random Vectors in Circular Region'});
filename = sprintf('test_data_distribution.jpg');
saveas(gcf, filename);

%% Predict using Already Trained Model
%% Visualize the Classifier
[predictedL, ~] = predict(rMdlLin, x');
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(x(1, :)) : d:max(x(1, :)),...
    min(x(2, :)) : d:max(x(2, :)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(rMdlLin, xGrid);
figure;
h = nan(3, 1); 
h(1:2) = gscatter(x(1, :).', x(2, :).', predictedL, 'rg', '+*');
hold on
h(3) = plot(x(1, rMdlLin.IsSupportVector).', x(2, rMdlLin.IsSupportVector).', 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');
legend(h, {'Predicted Gaussian','Predicted Random Vectors in Circular Region', 'Support Vectors'}, 'Location', 'southoutside');
axis equal
hold off
filename = sprintf('test_linear_svm_classification.jpg');
saveas(gcf, filename);

C = confusionmat(L, predictedL);
totalSamples = sum(C, 'all');
misclassifications = 0;
% compute misclassifications
for i = 1:2
    for j = 1:2
        if i ~= j
            misclassifications = misclassifications + C(i, j);
        end
    end
end
% percentage misclassifications
fprintf('Probability of Error: %.6f', (misclassifications / N));

%% Visualize the Classifier
[predictedL, ~] = predict(rMdlGau, x');
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(x(1, :)) : d:max(x(1, :)),...
    min(x(2, :)) : d:max(x(2, :)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(rMdlGau, xGrid);
figure;
h = nan(3, 1);
h(1:2) = gscatter(x(1, :).', x(2, :).', predictedL, 'rg', '+*');
hold on
h(3) = plot(x(1, rMdlGau.IsSupportVector).', x(2, rMdlGau.IsSupportVector).', 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');
legend(h, {'Predicted Gaussian','Predicted Random Vectors in Circular Region', 'Support Vectors'}, 'Location', 'southoutside');
axis equal
hold off
filename = sprintf('test_gaussian_svm_classification.jpg');
saveas(gcf, filename);

C = confusionmat(L, predictedL);
totalSamples = sum(C, 'all');
misclassifications = 0;
% compute misclassifications
for i = 1:2
    for j = 1:2
        if i ~= j
            misclassifications = misclassifications + C(i, j);
        end
    end
end
% percentage misclassifications
fprintf('Probability of Error: %.6f', (misclassifications / N));

