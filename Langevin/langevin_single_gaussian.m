%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mulitivariate gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc;

% define the mean and covariance
mu = [0; 0];
sigma = [1.0 0.0; 
         0.0 1.0];

% define grid range and resolution
x_range = -2:0.05:4;
y_range = -2:0.05:4;
[X, Y] = meshgrid(x_range, y_range);

% allocate Z for probability values
Z = zeros(size(X));

% evaluate PDF at each grid point
for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        x_vec = [X(i, j); Y(i, j)];
        Z(i, j) = normal_pdf(x_vec, mu, sigma);
    end
end

% Langevin initial condition
x0 = [0; 0]; % initial point

% plot 2D heat map
figure;
imagesc(x_range, y_range, Z);
xline(0); yline(0);
axis xy; % ensures y increases upwards
axis equal tight;
xlabel('x'); ylabel('y');
title('2D Heat Map of Multivariate Gaussian');
colorbar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% probability density function for normal distribution
function p = normal_pdf(x, mu, sigma)

    % compute the inverse and det of the covariance matrix
    inv_sigma = inv(sigma);
    det_sigma = det(sigma);

    % compute the difference
    diff = x - mu;

    % compute the probability density
    p = (1 / (2 * pi * sqrt(det_sigma))) * exp(-0.5 * diff' * inv_sigma * diff) ;
    
end
