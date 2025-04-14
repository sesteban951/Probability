%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Langevin dynamics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;

% Define mean and covariance
mu1 = [2;         
       2];         
sigma1 = [1.0 0.0;  
          0.0 0.5]; 
mu2 = [-2;         
        2]; 
sigma2 = [2.0 0.0;  
          0.0 1.0]; 
mu3 = [0;         
      -2]; 
sigma3 = [0.5 0.2;  
          0.2 1.0]; 
weights = [0.3; 0.4; 0.3]; % weights for the Gaussian components

% pack the means and covariances into arrays
n_dists = 3;
mu_list = zeros(2, 1, n_dists);
mu_list(:,:, 1) = mu1;
mu_list(:,:, 2) = mu2;
mu_list(:,:, 3) = mu3;

sigma_list = zeros(2, 2, n_dists);
sigma_list(:,:, 1) = sigma1;
sigma_list(:,:, 2) = sigma2;
sigma_list(:,:, 3) = sigma3;

% compute the mean of the means
mu_avg = zeros(2, 1);
sigma_avg = zeros(2, 2);
for i = 1:n_dists
    mu_avg = mu_avg + weights(i) * mu_list(:, :, i);
    sigma_avg = sigma_avg + weights(i) * sigma_list(:, :, i);
end

% Create a grid of (x, y) points
x_range = linspace(mu_avg(1) - 5*sqrt(sigma_avg(1,1)), mu_avg(1) + 5*sqrt(sigma_avg(1,1)), 50);
y_range = linspace(mu_avg(2) - 5*sqrt(sigma_avg(2,2)), mu_avg(2) + 5*sqrt(sigma_avg(2,2)), 50);
[X, Y] = meshgrid(x_range, y_range);

% Pack into points
pos = [X(:), Y(:)];

% Compute the Gaussian PDF
Z = zeros(size(pos, 1), 1);
for i = 1:size(pos, 1)
    
    % Compute the probability density for each point
    x = pos(i, :)';
    z = normal_pdf(x, mu_list, sigma_list, weights);
    
    % Reshape the result back to the grid
    Z(i) = z;
end

% Reshape Z to match the grid dimensions of X and Y
Z = reshape(Z, size(X));

% Plot the surface
figure;
surf(X, Y, Z, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); zlabel('Probability Density');
title('2D Gaussian Distribution');
colormap turbo;
colorbar;
view(45, 45); % 3D view angle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% probability density function for normal distribution
function p = normal_pdf(x, mu_list, sigma_list, weights)
    
    % number of distributions
    n_dists = size(mu_list, 3);

    % average out the PDFs
    p = 0;
    for i = 1:n_dists

        mu = mu_list(:, :, i);
        sigma = sigma_list(:, :, i);

        % compute the inverse and det of the covariance matrix
        inv_sigma = inv(sigma);
        det_sigma = det(sigma);

        % compute the difference
        diff = x - mu;

        % compute the probability density
        p_i = (1 / (2 * pi * sqrt(det_sigma))) * exp(-0.5 * diff' * inv_sigma * diff) ;
        p = p + weights(i) * p_i;
    end
end
