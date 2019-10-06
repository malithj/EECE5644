%% EECE5644 - Homework 2 - Question 3
close all;
numCases = 6;

for i = 1 : numCases
    index = (i - 1) * 3 + 1;
    data = cat(2, dataSet(:, index), dataSet(:, index + 1));
    labels = dataSet(:, index + 2);
    MdlLinear = fitcdiscr(data, labels);
    meandata = mean(data);
    predicted = predict(MdlLinear, data);
    figure;
    gscatter(data(:, 1), data(:, 2), labels,'krb','ov^',[],'off');
    hold on;
    legend('class 1', 'class 2');
    K = MdlLinear.Coeffs(1, 2).Const;  
    L = MdlLinear.Coeffs(1, 2).Linear;
    
    % Plot the discriminating boundary
    % K + [x1  x2]L = 0
    f = @(x1, x2) K + L(1) * x1 + L(2) * x2;
    h2 = fimplicit(f, [-4 8 -4 6]);
    h2.Color = 'r';
    h2.LineWidth = 2;
    h2.DisplayName = 'Boundary between Class 1 & Class 2';
    heading = sprintf("Case: %d, Fisher's Discriminant Analysis", i);
    title(heading);
    filename = sprintf('plot_lda_%d.jpg', i);
    saveas(gcf, filename);
end