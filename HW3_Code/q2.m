%% EECE5644 - Homework 3 - Question 2
clear all; close all; clc;

symbols = ['x', 'o', '+'];
m(:, 1) = [-1; 0]; Sigma(:, :, 1) = 0.1 * [10 -4;-4 5];   % mean and covariance of data pdf conditioned on label 1
m(:, 2) = [1; 0]; Sigma(:, :, 2) = 0.1 * [5 2;2 2];       % mean and covariance of data pdf conditioned on label 2
classPriors = [0.3, 0.7]; thr = [0, cumsum(classPriors)];
N = 999; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'brg';
rng(101); % set the seed

for l = 1:2
    indices = find(thr(l)<=u & u<thr(l+1)); % u precisely 1 not handled
    if l == 2 && sum(1 == u > 0)            % u precisely 1 handled by classifying as the last label
        indices_u_1 = find(1 == u);
        indices = [indices, indices_u_1];
    end
    L(1, indices) = l * ones(1, length(indices));
    x(:, indices) = mvnrnd(m(:,l), Sigma(:,:,l), length(indices))';
    gm = gmdistribution(m(:, l)', Sigma(:, :, l)');    % plot contours for better visualization
    gmPDF = @(x,y)reshape(pdf(gm,[x(:) y(:)]),size(x));
    f1 = fcontour(gmPDF,[-4 6], 'LineStyle', '--');
    hold on;
    figure(1), plot(x(1, indices), x(2, indices), strcat(symbols(l), colorList(l)), 'MarkerSize', 2); axis equal, hold on,
end

ylim([-3 3]);
xlim([-5 3]);
xlabel('Feature 1');
ylabel('Feature 2');
title(['Original Data']);
legend({'Class 1 Contours', 'Class 1 Points', 'Class 2 Contours', 'Class 2 Points'});
filename = sprintf('original_data.jpg');
saveas(gcf, filename); % plot saved

%% MAP Classifier
% define class conditional probability models
gm_1 = gmdistribution(m(:, 1).', Sigma(:, :, 1));
gm_2 = gmdistribution(m(:, 2).', Sigma(:, :, 2));
priors = [0.3, 0.7];

classifiedLabels = zeros(1, N);
for i = 1:N
    p_1 = pdf(gm_1, x(:, i).') * priors(1);
    p_2 = pdf(gm_2, x(:, i).') * priors(2);
    [max_probability, index] = max([p_1, p_2]);
    classifiedLabels(1, i) = index;
end

trueColor = 'g';
decisionColor = 'm';
% generate scatter plot with true labels and decision labels
for l= 1:2
    true_indices = find(L == l);
    decision_indices = find(classifiedLabels == l);
    correct = intersect(true_indices, decision_indices);
    errors = setdiff(decision_indices, true_indices);
    figure(2), plot(x(1, true_indices), x(2, true_indices), strcat(symbols(l), trueColor), 'MarkerSize', 5); axis equal, hold on,
    figure(2), plot(x(1, errors), x(2, errors), strcat(symbols(l), decisionColor), 'MarkerSize', 2); hold on,
    ylim([-3 3]);
    xlim([-5 3]);
    hold on;
end

xlabel('Feature 1');
ylabel('Feature 2');
title(['MAP Classifier Performance (True vs Predicted)']);
legend({'Classified Correctly C1', 'Misclassified as C1', 'Classified Correctly C2', 'Misclassified as C2'});
filename = sprintf('classified_map_data_class.jpg');
saveas(gcf, filename);

% MAP Performance Statistics
C = confusionmat(L, classifiedLabels);
% display confusion matrix
display(C);
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
% display number of misclassifications
display(misclassifications);
% percentage misclassifications
fprintf('MAP – Probability of Error: %.6f', (misclassifications / 10000));


%% Fisher's LDA
MdlLinear = fitcdiscr(x', L');
predicted = predict(MdlLinear, x');
figure(3);
gscatter(x(1, :), x(2, :), L,'krb','ov^',[],'off');
hold on;
legend('class 1', 'class 2');
KFisher = MdlLinear.Coeffs(1, 2).Const;  
LFisher = MdlLinear.Coeffs(1, 2).Linear;

% Plot the discriminating boundary
% KFisher + [x1  x2]LFisher = 0
f = @(x1, x2) KFisher + LFisher(1) * x1 + LFisher(2) * x2;
h2 = fimplicit(f, [-4 8 -4 6]);
h2.Color = 'r';
h2.LineWidth = 2;
h2.DisplayName = 'Boundary between Class 1 & Class 2';
heading = sprintf("Fisher's Discriminant Analysis");
title(heading);
xlabel('Feature 1');
ylabel('Feature 2');
filename = sprintf('plot_lda.jpg');
saveas(gcf, filename);

% Fisher's LDA Classifier
classifiedLabels = zeros(1, N);
for i = 1:N
    discVal = f(x(1, i), x(2, i));
    if discVal >= 0
        classifiedLabels(1, i) = 1;
    else
         classifiedLabels(1, i) = 2;
    end
end

trueColor = 'g';
decisionColor = 'm';
% generate scatter plot with true labels and decision labels
for l= 1:2
    true_indices = find(L == l);
    decision_indices = find(classifiedLabels == l);
    correct = intersect(true_indices, decision_indices);
    errors = setdiff(decision_indices, true_indices);
    figure(4), plot(x(1, true_indices), x(2, true_indices), strcat(symbols(l), trueColor), 'MarkerSize', 5); axis equal, hold on,
    figure(4), plot(x(1, errors), x(2, errors), strcat(symbols(l), decisionColor), 'MarkerSize', 2); hold on,
    ylim([-3 3]);
    xlim([-5 3]);
    hold on;
end

h2 = fimplicit(f, [-4 8 -4 6]);
h2.Color = 'r';
h2.LineWidth = 2;
h2.DisplayName = 'Boundary between Class 1 & Class 2';
xlabel('Feature 1');
ylabel('Feature 2');
title(['Fisher''s LDA Classifier Performance (True vs Predicted)']);
legend({'Classified Correctly C1', 'Misclassified as C1', 'Classified Correctly C2', 'Misclassified as C2'});
filename = sprintf('classified_lda_data_class.jpg');
saveas(gcf, filename);

% Fisher's LDA Performance Statistics
C = confusionmat(L, classifiedLabels);
% display confusion matrix
display(C);
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
% display number of misclassifications
display(misclassifications);
% percentage misclassifications
fprintf('LDA – Probability of Error: %.6f', (misclassifications / 10000));

%% Logistic Regression
[B, dev, stats] = mnrfit(x', L);
lgr = @(x1, x2) B(1) + B(2) * x1 + B(3) * x2;

% Logistic Regression Classifier
classifiedLabels = zeros(1, N);
for i = 1:N
    discVal = lgr(x(1, i), x(2, i));
    if discVal >= 0
        classifiedLabels(1, i) = 1;
    else
        classifiedLabels(1, i) = 2;
    end
end

trueColor = 'g';
decisionColor = 'm';
% generate scatter plot with true labels and decision labels
for l= 1:2
    true_indices = find(L == l);
    decision_indices = find(classifiedLabels == l);
    correct = intersect(true_indices, decision_indices);
    errors = setdiff(decision_indices, true_indices);
    figure(5), plot(x(1, true_indices), x(2, true_indices), strcat(symbols(l), trueColor), 'MarkerSize', 5); axis equal, hold on,
    figure(5), plot(x(1, errors), x(2, errors), strcat(symbols(l), decisionColor), 'MarkerSize', 2); hold on,
    ylim([-3 3]);
    xlim([-5 3]);
    hold on;
end

h2 = fimplicit(lgr, [-4 8 -4 6]);
h2.Color = 'r';
h2.LineWidth = 2;
h2.DisplayName = 'Boundary between Class 1 & Class 2';
xlabel('Feature 1');
ylabel('Feature 2');
title(['Logistic Regression Classifier ', newline, 'Performance (True vs Predicted)']);
legend({'Classified Correctly C1', 'Misclassified as C1', 'Classified Correctly C2', 'Misclassified as C2'});
filename = sprintf('classified_lgr_data_class.jpg');
saveas(gcf, filename);

% LGR Performance Statistics
C = confusionmat(L, classifiedLabels);
% display confusion matrix
display(C);
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
% display number of misclassifications
display(misclassifications);
% percentage misclassifications
fprintf('Logistic Regression – Probability of Error: %.6f', (misclassifications / 10000));
