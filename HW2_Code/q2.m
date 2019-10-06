%% EECE5644 - Homework 2 - Question 2
clear all; close all; clc;
N = 400;
dataSet = generateDataAndPlots(N);

function out = generateDataAndPlots(N)
    numCases = 6;
    dataSet = [];
    mu_array = cat(3, [0 0; 3 3], [0 0; 3 3], [0 0; 2 2], [0 0; 3 3], [0 0; 3 3], [0 0; 2 2]);
    sigma_array = cat(4, cat(3, eye(2), eye(2)), cat(3, [3 1; 1 0.8], [3 1; 1 0.8]), cat(3, [2 0.5; 0.5 1], [2 -1.9; -1.9 5]), cat(3, eye(2), eye(2)), cat(3, [3 1; 1 0.8], [3 1; 1 0.8]), cat(3, [2 0.5; 0.5 1], [2 -1.9; -1.9 5]));
    p_array = cat(3, ones(1, 2)/2, ones(1, 2)/2, ones(1, 2)/2, [0.05 0.95], [0.05 0.95], [0.05 0.95]);
    
    % loop through the iterations
    for i = 1 : numCases
        mu = mu_array(:, :, i);
        sigma = sigma_array(:, :, :, i);
        p = p_array(:, :, i);
        gm = gmdistribution(mu, sigma, p);
        
        % Generate random variates 
        rng('default'); % For reproducibility
        [Y, compIdx] = random(gm, N);
        
        % Create array with original class labels
        X = cat(2, Y, compIdx);
        dataSet = cat(4, dataSet, X);
        classOneDataIdx = [X(:, 3) == 1];
        classTwoDataIdx = [X(:, 3) == 2];
        classOneData = Y(classOneDataIdx, :);
        classTwoData = Y(classTwoDataIdx, :);
        figure;
        
        % Generate contour plot
        scatter(classOneData(:,1),classOneData(:,2), 15, 'o', 'b', 'filled');
        hold on;
        scatter(classTwoData(:,1),classTwoData(:,2), 15, 'd', 'r', 'filled');
        hold on;
        gmPDF = @(x,y)reshape(pdf(gm,[x(:) y(:)]),size(x));
        f1 = fcontour(gmPDF,[-4 6], 'LineStyle', '--');
        hold on;
        set(get(get(f1(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        legend('class 1', 'class 2');
        xlabel('x_{1}','FontSize',14);
        ylabel('x_{2}','FontSize',14);
        title('Contour lines of pdf and Random Variates', 'FontSize', 16);
        filename = sprintf('plot_case_%d.jpg', i);
        saveas(gcf, filename);
        
        % Maximum A Posteriori
        P = posterior(gm, Y);
        PCat = cat(2, P, P(:, 1) ./ P(:, 2) >= 1);
        % Inferred class labels
        Z = cat(2, PCat(:, 1 : 2), 3 - (PCat(:, 3) + 1));
        classOneIdx = [PCat(:, 3) == 1];
        classTwoIdx = [PCat(:, 3) == 0];
        classOne = Y(classOneIdx, :);
        classTwo = Y(classTwoIdx, :);
        
        % class performance 
        fprintf('case_%d\n', i);
        cp = classperf(X(:, 3), Z(:, 3))
        
        % scatter plot
        figure;
        scatter(classOne(:,1),classOne(:,2), 10, 'o', 'b', 'filled');
        hold on;
        scatter(classTwo(:,1),classTwo(:,2), 10, 's', 'r', 'filled');
        legend('class 1', 'class 2');
        xlabel('x_{1}','FontSize', 14);
        ylabel('x_{2}','FontSize', 14);
        title('Inferred Decision Labels using MAP');
        filename = sprintf('plot_case_map_%d.jpg', i);
        saveas(gcf, filename);
    end
    out = dataSet;
end






