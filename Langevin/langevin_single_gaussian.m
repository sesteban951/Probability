%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mulitivariate gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc;

% define the mean and covariance
% mu = [1; 0];
% sigma = [1.0 0.0; 
%          0.0 1.0];
mu_xlim = 2;
mu_ylim = 2;
mu = [rand(1)*2*mu_xlim - mu_xlim; 
      rand(1)*2*mu_ylim - mu_ylim];
sigma_sqrt = rand(2, 2);
sigma = sigma_sqrt * sigma_sqrt';

% define grid range and resolution
discretization = 0.05;
x_range = mu(1)-1.5:discretization:mu(1)+1.5;
y_range = mu(2)-1.5:discretization:mu(2)+1.5;
[X, Y] = meshgrid(x_range, y_range);

% allocate Z for probability values
Z = zeros(size(X));
Z_grad = zeros(size(X, 1), size(X, 2), 2);

% evaluate PDF at each grid point
for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        
        % evluate at this point
        x_vec = [X(i, j); Y(i, j)];
        
        % compute the probability density
        Z_point = normal_pdf(x_vec, mu, sigma);
        Z(i, j) = Z_point;

        % compute the gradient of the probability density
        Z_point_grad = normal_pdf_grad(x_vec, mu, sigma);
        Z_point_grad = Z_point_grad / norm(Z_point_grad); % normalize the gradient
        Z_grad(i, j, :) = Z_point_grad;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LANGEVIN DYNAMICS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Langevin initial condition
x0 = [0; 0]; % initial point

% Langevin parameters
alpha = 0.001;                 % time step
num_steps = 500;          % number of steps
x = zeros(num_steps, 2);   % initialize Langevin trajectory
x(1, :) = x0';     % set initial condition

% simulate the Langevin dynamics (realized with Euler-Maruyama)
for i = 2:num_steps
    % compute the gradient of the probability density at the current point
    xk = x(i-1, :)';
    grad = log_normal_pdf_grad(xk, mu, sigma);

    % get a noise vector from standard normal distribution
    noise_vec = randn(2, 1);

    % take a Langevin step
    xk = xk + alpha * grad + sqrt(2 * alpha) * noise_vec;

    % save the new point
    x(i, :) = xk';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot 2D heat map of the probability density
figure;
hold on;
imagesc(x_range, y_range, Z);
xline(0); yline(0);
axis xy; % ensures y increases upwards
axis equal tight;
xlabel('x'); ylabel('y');
title('2D Heat Map of Multivariate Gaussian');
colorbar;

% plot the mean 
plot(mu(1), mu(2), 'k.', 'MarkerSize', 30);

% plot the gradient field
quiver(X, Y, Z_grad(:, :, 1), Z_grad(:, :, 2), 'r');

% plot the Langevin trajectory
plot(x(:, 1), x(:, 2), 'g-', 'LineWidth', 2);
plot(x(1, 1), x(1, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'LineWidth', 2);
plot(x(end, 1), x(end, 2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'LineWidth', 2);

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

% gradient of the normal pdf
function grad = normal_pdf_grad(x, mu, sigma)
 
    % compute the inverse and det of the covariance matrix
    inv_sigma = inv(sigma);
    det_sigma = det(sigma);

    % compute the difference
    diff = x - mu;

    % compute the gradient
    grad = -inv_sigma * diff / (2 * pi * sqrt(det_sigma));
end

% gradient of the log normal pdf
function grad = log_normal_pdf_grad(x, mu, sigma)
    % comkpute the inverse of the covariance matrix
    inv_sigma = inv(sigma);

    % compute the gradient
    grad = -inv_sigma * (x - mu);
end