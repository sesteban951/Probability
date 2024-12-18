%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sampling Optimization on S^1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc;

% initial distribution
mu = 5*pi/4;
sigma = 0.1;

% goal point
p = [1.5, 1.5];

% CE-M points
N = 10;
N_elite = 2;
max_iters = 100;

% plot some stuff
figure;
hold on; axis equal;
xline(0); yline(0);

% plot the S1 manifold
N = 200;
theta = linspace(0, 2*pi, N);
S1 = [cos(theta); sin(theta)];
plot(S1(1,:), S1(2,:), 'k');
xlabel('x'); ylabel('y');

% plot the star point
plot(p(1), p(2), 'bp', 'MarkerSize', 10, 'LineWidth', 2);

x_S1 = pi/4
V = sample_tangent_space(x_S1, 0.1, 100);
plot(V(:,1), V(:,2), 'r.', 'MarkerSize', 15, 'LineWidth', 2);
plot(cos(x_S1), sin(x_S1), 'gp', 'MarkerSize', 10, 'LineWidth', 2);

% cost function for optimization
function [X_sorted, J_sorted] = cost_function(X, p)

    % turn into cartesian coordinates
    X_cart = [cos(X), sin(X)];

    % compute the distance from each point to the p
    J = zeros(length(X), 1);
    for i = 1:length(X)
        J(i) = norm(X_cart(i,:) - [p(1), p(2)]);
    end

    % sort the points by distance
    [J_sorted, idx] = sort(J, "ascend");
    X_sorted = X(idx);
end

% generate samples on tangent space
% V in T_x_S^1
function V = sample_tangent_space(x_S1, sigma, N)

    % compute the cartesian point on the S1
    x = [cos(x_S1), sin(x_S1)];

    % generate samples from normal distribution
    v = normrnd(0, sigma, N, 1);

    % compute the random vectors
    V = zeros(N, 2);
    for i = 1:N
        V(i, :) = [x(1), x(2)] + [v(i), -v(i) * (x(1)/x(2))];
    end
end

% given a point on the S1, find the riemannian metric
function d = riemannian_metric(y1, y2)
    
    % in this case, it's the arc distance (abs val of difference)
    p1 = [cos(y1), sin(y1)];
    p2 = [cos(y2), sin(y2)];

    % compute the angle between the vectors
    d = acos(dot(p1, p2));

end
