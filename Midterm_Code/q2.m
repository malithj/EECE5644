%% EECE5644 - MidTerm - Question 2
clear all; close all; clc;
n_sigma = 0.3;
n_mean = 0;
x_sigma = 0.25;
y_sigma = 0.25;
low = 0;
high = 1;
rng(50);  % set the seed for consistency

radius_of_true_point = 1 * sqrt(rand(1, 1));
theta_of_true_point = 2 * pi * rand(1, 1);

x_true = radius_of_true_point * cos(theta_of_true_point);
y_true = radius_of_true_point * sin(theta_of_true_point);

% iterate through k
for K=1:4
    n = n_mean + n_sigma * randn(1, K);
    
    % generate k landmarks 
    landmarks = [cos(2 * pi * rand(1, K)); sin(2 * pi * rand(1, K))];
    range_measurements = vecnorm(landmarks - [x_true, y_true].') + n;
    while sum(range_measurements < 0) > 0
        n_noise = n_mean + n_sigma * randn(1, sum(range_measurements < 0));
        indices = find(range_measurements < 0);
        range_measurements(indices) = range_measurements(indices) - n(indices) + n_noise;
    end
    
    x_n = 5;
    y_n = 5;
    result = zeros(x_n, y_n);
    
    % MAP
    for x = linspace(-2, 2, x_n)
        for y = linspace(-2, 2, y_n)
            map = (-1 / (2 * n_sigma^2)) * sum(range_measurements - vecnorm([x, y].' - landmarks)) - (1 / 2) * ((x ^ 2 / x_sigma ^ 2) + (y ^ 2 / y_sigma ^ 2));
            result(x + 3, y + 3) = map;   
        end
    end 
     
    figure
    [X, Y] = meshgrid(linspace(-2, 2, x_n), linspace(-2, 2, y_n));
    minx = -50;
    maxx = 0;
    levels =  minx:2.5:maxx;
    contour(X, Y, result, levels, 'Showtext', 'on');
    hold on;
    plot(x_true, y_true, '+', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    hold on;
    plot(landmarks(1,:), landmarks(1,:), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    xlabel('x');
    ylabel('y');
    title('MAP Estimate of [x, y] Given Range Measurements');
    legend({'MAP', 'True Position', 'Reference Points'});
    filename = sprintf('q2_K_%d.jpg', K);
    saveas(gcf, filename);
end