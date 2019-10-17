%% EECE5644 - MidTerm - Question 3
clear all; close all; clc;
low = -1;
high = 1;
N = 10;
v_sigma = 3.0;       % choose sigma in a way which will increase noise (standard distribution sigma: 1)

% choose real roots to be -1, 0, 1. i.e. x^3 - x is the expression.
W_true = [1, 0, -1, 0];

% when gamma is large, the prior probability component in MAP approaches zero
% compute MAP estimates based on gamma
start = -5; stop = 5;
totalRuns = 100;
squared_error = zeros(stop - start + 1, totalRuns);
for gamma = logspace(start, stop, stop - start + 1)
    for run = 1:totalRuns
        % generate all data in a random manner
        % generate x values from uniform distribution
        x = low + (high - low).* rand(1, N);
        % generate v array
        v_mean = 0;
        v = v_mean + v_sigma * randn(1, N);

        % generate the y dataset
        y = x.^3 - x + v;

        % define the equations to solve param vector
        syms a b c d;
        eqd = a * (sum(x.^3)) + b * (sum(x.^2)) + c * (sum(x)) + d * ((1 / (gamma^2)) + (1 / v_sigma^2) * N) == sum(y);
        eqc = a * (sum(x.^4)) + b * (sum(x.^3)) + d * (sum(x)) + c * ((1 / (gamma^2)) + (1 / v_sigma^2) * (sum(x.^2))) == sum(x.*y);
        eqb = a * (sum(x.^5)) + c * (sum(x.^3)) + d * (sum(x.^2)) + b * ((1 / (gamma^2)) + (1 / v_sigma^2) * (sum(x.^4))) == sum((x.^2).*y);
        eqa = b * (sum(x.^5)) + c * (sum(x.^4)) + d * (sum(x.^3)) + a * ((1 / (gamma^2)) + (1 / v_sigma^2) * (sum(x.^6))) == sum((x.^3).*y);
        sol = solve([eqa, eqb, eqc, eqd], [a, b, c, d]);
        solution = [vpa(sol.a), vpa(sol.b), vpa(sol.c), vpa(sol.d)];

        % L2 norm between two vectors
        norm_ = norm(solution - W_true)^2;
        iteration = stop + log10(gamma) + 1;
        squared_error(iteration, run) = norm_;
    end
end

% generate required outputs
maximum_ = max(squared_error.');
minimum_ = min(squared_error.');
median_ = median(squared_error.');
prc_25 = prctile(squared_error.', 25);
prc_75 = prctile(squared_error.', 75);


% generate matlab plot
figure;
plot(logspace(start, stop, stop - start + 1), maximum_);
hold on;
plot(logspace(start, stop, stop - start + 1), minimum_);
hold on;
plot(logspace(start, stop, stop - start + 1), median_);
hold on;
plot(logspace(start, stop, stop - start + 1), prc_25);
hold on;
plot(logspace(start, stop, stop - start + 1), prc_75);
xlim([10^(start) 10^(stop)]);
title('Squared Error of MAP Estimate');
xlabel('\gamma');
ylabel('Statistic');
legend({'maximum', 'minimum', 'median', 'percentile 25%', 'percentile 75%'});
set(gca, 'XScale', 'log');
filename = sprintf('q3.jpg');
saveas(gcf, filename);
