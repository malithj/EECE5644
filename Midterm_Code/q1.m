%% EECE5644 - MidTerm - Question 1
clear all; close all; clc;

symbols = ['x', 'o', '+'];
m(:, 1) = [-1; 0]; Sigma(:, :, 1) = 0.1 * [10 -4;-4,5];   % mean and covariance of data pdf conditioned on label 3
m(:, 2) = [1; 0]; Sigma(:, :, 2) = 0.1 * [5 0;0,2];       % mean and covariance of data pdf conditioned on label 2
m(:, 3) = [0; 1]; Sigma(:, :, 3) = 0.1 * eye(2);          % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15, 0.35, 0.5]; thr = [0, cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'brg';
rng(101); % set the seed

for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % u precisely 1 not handled
    if l == 3 && sum(1 == u > 0)            % u precisely 1 handled by classifying as the last label
        indices_u_1 = find(1 == u);
        indices = [indices, indices_u_1];
    end
    L(1, indices) = l * ones(1, length(indices));
    x(:, indices) = mvnrnd(m(:,l), Sigma(:,:,l), length(indices))';
    figure(1), plot(x(1, indices), x(2, indices), strcat(symbols(l), colorList(l)), 'MarkerSize', 2); axis equal, hold on,
end

ylim([-3 3]);
xlim([-5 3]);
xlabel('Feature 1');
ylabel('Feature 2');
title(['Original Data']);
legend({'Class 1', 'Class 2', 'Class 3'});
filename = sprintf('original_data.jpg');
saveas(gcf, filename);

% count the number of points generated in each class
class = zeros(1, 3);
for i = 1:3
    class(1, i) = sum(L == i);
end

% display the class count
display(class);

% define class conditional probability models
gm_1 = gmdistribution(m(:, 1).', Sigma(:, :, 1));
gm_2 = gmdistribution(m(:, 2).', Sigma(:, :, 2));
gm_3 = gmdistribution(m(:, 3).', Sigma(:, :, 3));
priors = [0.15, 0.35, 0.5];

classifiedLabels = zeros(1, N);
for i = 1:N
    p_1 = pdf(gm_1, x(:, i).') * priors(1);
    p_2 = pdf(gm_2, x(:, i).') * priors(2);
    p_3 = pdf(gm_3, x(:, i).') * priors(3);
    [max_probability, index] = max([p_1, p_2, p_3]);
    classifiedLabels(1, i) = index;
end

C = confusionmat(L, classifiedLabels);

% display confusion matrix
display(C);

totalSamples = sum(C, 'all');
misclassifications = 0;

% compute misclassifications
for i = 1:3
    for j = 1:3
        if i ~= j
            misclassifications = misclassifications + C(i, j);
        end
    end
end

% display number of misclassifications
display(misclassifications);

% percentage misclassifications
fprintf('Probability of Error: %.6f', (misclassifications / 10000));

trueColor = 'g';
decisionColor = 'm';
% generate scatter plot with true labels and decision labels
for l= 1:3
    true_indices = find(L == l);
    decision_indices = find(classifiedLabels == l);
    correct = intersect(true_indices, decision_indices);
    errors = setdiff(decision_indices, true_indices);
    figure(l + 1), plot(x(1, true_indices), x(2, true_indices), strcat(symbols(l), trueColor), 'MarkerSize', 5); axis equal, hold on,
    figure(l + 1), plot(x(1, errors), x(2, errors), strcat(symbols(l), decisionColor), 'MarkerSize', 2); hold on,
    ylim([-3 3]);
    xlim([-5 3]);
    xlabel('Feature 1');
    ylabel('Feature 2');
    title(['Class ', num2str(l), ' Performance (True vs Predicted)']);
    legend({'Classified Correctly', ['Misclassified as C', num2str(l)]});
    filename = sprintf('classified_data_class_%d.jpg', l);
    saveas(gcf, filename);
end
