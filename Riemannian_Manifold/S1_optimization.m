%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling Optimization on S^1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc;

% CE-M points
N = 50;
N_elite = 2;
max_iters = 10;
epsilon = 1E-9;

% goal point
p = [1, 0.1];

% initial distribution
mu = pi;
sigma2 = 1.5^2;
var_sacling = 50; % NOTE: this helps with not collapsing to zero too fast

% iterate until convergence
iter = 0;
while (iter < max_iters) && (sigma2 > epsilon)

    % Obtain N samples from the current sampling distribution
    V = normrnd(0, sigma2, N, 1);
    X = Exp_map(V, mu);

    % store the results
    X_hist(:,:,iter+1) = X;
    mu_hist(iter+1) = mu;

    % Evaluate the objective function at the sampled points
    F = cost_function(X, p);

    % Sort the samples based on the objective function
    [~, idx] = sort(F, 'ascend');  % smallest to biggest
    X_sorted = X(idx);

    % Update the distribution parameters from elite samples
    X_elite = X_sorted(1:N_elite);
    V_elite = Log_map(X_elite, mu);
    mu = mu + mean(V_elite);
    sigma2 = var_sacling * var(V_elite);

    % increment the iteration count
    iter = iter + 1;
end

% covert to cartesian points
size_X = size(X_hist);
size_mu = size(mu_hist);
for i = 1:size_X(3)
    for j = 1:size_X(1)
        X_hist_cart(j,:,i) = [cos(X_hist(j,i)), sin(X_hist(j,i))];
    end
    mu_hist_cart(i,:) = [cos(mu_hist(i)), sin(mu_hist(i))];
end

% display the results
fprintf('Optimization finished in %d iterations\n', iter);
fprintf('The final mean is %.3f and the final variance is %.3f\n', mu, sigma2);

% plot some stuff
figure;
hold on; axis equal;
xline(0); yline(0);

% plot the S1 manifold
N = 500;
theta = linspace(0, 2*pi, N);
S1 = [cos(theta); sin(theta)];
plot(S1(1,:), S1(2,:), 'k');
xlabel('x'); ylabel('y');

% plot the star point
plot(p(1), p(2), 'bp', 'MarkerSize', 10, 'LineWidth', 2);
plot([0, p(1)], [0, p(2)], 'k--');

% plot the cost function on the manifold
% F = cost_function(theta, p);
% F_manif = 

% plot the final mean
for i = 1:size_mu(2)
    
    % plot the distribution
    distirbution = plot(X_hist_cart(:,1,i), X_hist_cart(:,2,i), 'r.', 'MarkerSize', 25);
    mean = plot(mu_hist_cart(i,1), mu_hist_cart(i,2), 'gp', 'MarkerSize', 10, 'LineWidth', 2);

    msg = sprintf('Iteration: %d, mu: %.2f, sigma2: %.2f', i, mu_hist(i), sigma2);
    title(msg);
    
    pause(1.0);

    % remove the previous plot
    if i < size_mu(2)
        delete(distirbution);
        delete(mean);
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cost function for optimization (simple Eucledian distance)
function J = cost_function(X, p)

    % turn into cartesian coordinates
    X_cart = [cos(X), sin(X)];

    % compute the distance from each point to the p
    J = zeros(length(X), 1);
    for i = 1:length(X)
        J(i) = norm(X_cart(i,:) - [p(1), p(2)], 2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% take elemnts from V in T_xS^1 to X in S^1
function X = Exp_map(V, x)

    % counter-clockwise is positive
    % (adding geodesic distance to point p)
    X_exp = x + V; 

    % wrap around 
    X = mod(X_exp, 2*pi);
end

% take elements from X in S^1 to V in T_xS^1
function V = Log_map(X, x)

    % counter-clockwise is positive
    % (subtracting geodesic distance to point p)
    V = X - x;
end

% given two points x1, x2 in S1, compute the riemannian metric
function d = riemannian_metric(x1, x2)
    
    % need to account for wrapping
    d1 = abs(x1 - x2);
    d2 = 2*pi - abs(x1 - x2);
    d = min(d1, d2);

end
