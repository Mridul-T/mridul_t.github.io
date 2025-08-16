% RANDOM FIELD VISCOSITY
clear all;
close all;
% Check if the command-line arguments are passed
% if exist('a', 'var') == 0 || exist('b', 'var') == 0
%     % Prompt user for inputs
%     a = input('Enter the value for a: ');
%     b = input('Enter the value for b: ');
% end

a=1;
b=2501;

k = 1;
con = 1;
particles_per_unit = 50 * k;
dx = con / particles_per_unit;
ratio = 1.4;
h = ratio * dx;
c_ = 2 * h;
q = 30;
max_ord = 3;
n_samples = 10000;



J = particles_per_unit^2; % Total number of particles
rho = 1000 * ones(J, 1); % Density for steel
mass = rho * dx^2;
rho0 = rho;

P = 1;

% Create meshgrid for particle positions
x_values = linspace(0, 1 * con, particles_per_unit);
y_values = linspace(0, 1 * con, particles_per_unit);
[X, Y] = meshgrid(x_values, y_values);

% Flatten the meshgrid arrays
X_flat = X(:);
Y_flat = Y(:);

% Combine into a single array of 2D positions
positions = [X_flat, Y_flat];

% Construct the covariance kernel
sigma = 0.001;
l = 0.01;
% Construct the covariance kernel
K = squared_exponential_kernel_2d(positions, positions, sigma, l);

% Compute the eigenvalues and eigenvectors of the covariance matrix
[eigenvectors, D] = eig(K);
eigenvalues = diag(D);

% Sort the eigenvalues and eigenvectors in descending order
[eigenvalues_sorted, idx] = sort(eigenvalues, 'descend');
eigenvectors_sorted = eigenvectors(:, idx);

coords = zeros(P, J, 2);
coords(1, :, :) = positions;


% Parallel computation
num_cores = feature('numcores');

parpool(num_cores); % Start parallel pool

start_time = tic;

% Preallocate the results as a struct array instead of a cell
results = zeros(b - a + 1, 2, 2500, 301); % Properly preallocate with correct dimensions

parfor idx = 1:(b - a + 1)
    i = a + idx - 1; % Map idx to 'i'
    results(idx,:,:,:) = reshape(solver_wrapper(X, Y, particles_per_unit, J, i, eigenvalues_sorted, eigenvectors_sorted), [1, 2, 2500, 301]); % Explicitly match dimensions
end

elapsed_time = toc(start_time);

fprintf('Total time taken %.2f mins\n', elapsed_time / 60);

% Convert results (4D array) to float32
results_single = single(results);

% Save the converted results to a file
save('results_mcs3_test1.mat', 'results_single', '-v7.3');

% Solver wrapper function
function result = solver_wrapper(X,Y, particles_per_unit, J, i, eigenvalues_sorted, eigenvectors_sorted)
    u_init = cal_u(X, Y, particles_per_unit, J, i);
    fprintf('Starting the %dth iteration\n', i);
    result = solver(u_init, eigenvalues_sorted, eigenvectors_sorted, i); % Assuming solver2 is a MATLAB function
    fprintf('Done with %dth iteration\n', i);
end

% Function to calculate initial velocity
function u0 = cal_u(X, Y, particles_per_unit, J, i)
    u0_temp = zeros(2, particles_per_unit, particles_per_unit);
    u0_temp(1, :, :) = 0.25 * sin(2 * pi * X) .* sin(2 * pi * Y);
    u0_temp(2, :, :) = -0.1 * sin(2 * pi * X) .* sin(2 * pi * Y);
    
    u0 = zeros(2, J);
    u0(1, :) = reshape(u0_temp(1, :, :), [1, J]);
    u0(2, :) = reshape(u0_temp(2, :, :), [1, J]);
end


function ut = solver(u_init,eigenvalues_sorted, eigenvectors_sorted, i)
    rng(i); % Seed the random number generator
    
    % Parameters
    k = 1;
    con = 1;
    particles_per_unit = 50 * k;
    dx = con / particles_per_unit;
    ratio = 1.4;
    h = ratio * dx;
    
    J = particles_per_unit^2; % Total number of particles
    rho = 1000 * ones(J, 1);  % Density for steel
    mass = rho * dx^2;
    
    T = 0.30;    % Total time of integration
    dt = 0.001;  % Time step
    N = floor(T / dt);

    % Create a meshgrid for particle positions
    x_values = linspace(0, 1 * con, particles_per_unit);
    y_values = linspace(0, 1 * con, particles_per_unit);
    [X, Y] = meshgrid(x_values, y_values);

    % Flatten the meshgrid for 1D coordinates
    X_flat = X(:);
    Y_flat = Y(:);
    coords = [X_flat, Y_flat];
    
    % Central Difference Function
    function [ux, uy] = CD(ud, dx)
    
        [nx, ny] = size(ud);
        ux = zeros(size(ud));
        uy = zeros(size(ud));
        
        % Compute ux
        for i = 2:nx-1
            ux(i, :) = (ud(i+1, :) - ud(i-1, :)) / (2 * dx);
        end
        ux(1, :) = (-3/2 * ud(1, :) + 2 * ud(2, :) - ud(3, :) / 2) / dx;
        ux(nx, :) = (3/2 * ud(nx, :) - 2 * ud(nx-1, :) + ud(nx-2, :) / 2) / dx;
        
        % Compute uy
        for j = 2:ny-1
            uy(:, j) = (ud(:, j+1) - ud(:, j-1)) / (2 * dx);
        end
        uy(:, 1) = (-3/2 * ud(:, 1) + 2 * ud(:, 2) - ud(:, 3) / 2) / dx;
        uy(:, ny) = (3/2 * ud(:, ny) - 2 * ud(:, ny-1) + ud(:, ny-2) / 2) / dx;
    end

    % Initialize velocity field (2D)
    ut = zeros(2, J, N+1);
    ut(1, :, 1) = u_init(1, :);
    ut(2, :, 1) = u_init(2, :);
    
    % Boundary mask
    x_mask = (coords(:, 1) == 0) | (coords(:, 1) == 1);
    y_mask = (coords(:, 2) == 0) | (coords(:, 2) == 1);
    mask = (x_mask | y_mask);
    
    q = 5; % Number of components
    w = randn(q, 1); % Example coefficients
    viscosity = cal_f(w, eigenvalues_sorted, eigenvectors_sorted, q);
    sum(viscosity,1)
    viscosity = reshape(viscosity,[particles_per_unit,particles_per_unit]);
    
    % Time-stepping loop
    for n = 2:N+1
        % Example usage
        u = reshape(ut(:, :, n-1), [2, particles_per_unit, particles_per_unit]);

        % Compute first derivatives
        u1 = squeeze(u(1, :, :)); % Remove singleton dimension for u_x
        u2 = squeeze(u(2, :, :)); % Remove singleton dimension for u_y

        [ux_d1, ux_d2] = CD(u1, dx); % For u_x
        [uy_d1, uy_d2] = CD(u2, dx); % For u_y

        % Compute second derivatives
        [ux_ddx, ~] = CD(ux_d1, dx);
        [~, ux_ddy] = CD(ux_d2, dx);
        [uy_ddx, ~] = CD(uy_d1, dx);
        [~, uy_ddy] = CD(uy_d2, dx);

        % RHS computation
        
        rhsx = u1 .* ux_d1 + u2 .* ux_d2 - viscosity .* (ux_ddx + ux_ddy);
        rhsy = u1 .* uy_d1 + u2 .* uy_d2 - viscosity .* (uy_ddx + uy_ddy);

        % Update velocity and ensure the shape matches [1, J]
        ut(1, :, n) = reshape(u1 - rhsx * dt, [1, J]);
        ut(2, :, n) = reshape(u2 - rhsy * dt, [1, J]);

        % Apply boundary conditions (set velocity to 0 at mask)
        ut(:, mask, n) = 0;

        % Divergence check
        if any(ut(:, :, n) > 1e3, 'all')
            fprintf('ERROR: DIVERGING SOLUTION\n');
            break;
        end
    end
    % Return the final result
    ut = ut;
end



% Define the squared exponential kernel function for 2D
function K = squared_exponential_kernel_2d(pos1, pos2, sigma, l)
    % Compute pairwise squared distances
    [M1, M2] = ndgrid(1:size(pos1, 1), 1:size(pos2, 1));
    distances_sq = sum((pos1(M1, :) - pos2(M2, :)).^2, 2);
    distances_sq = reshape(distances_sq, size(M1));
    % Compute the kernel
    K = sigma * exp(-0.5 * distances_sq / l^2);
end

% Define the coefficient function u
function f = cal_f(w, eigenvalues_sorted, eigenvectors_sorted, q)
    f = 0.05;
    f = f + eigenvectors_sorted(:, 1:q) * (sqrt(eigenvalues_sorted(1:q)) .* w(:));
end
