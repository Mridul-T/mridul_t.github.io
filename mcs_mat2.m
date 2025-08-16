% MATLAB Equivalent Code
clear all;
close all;
% Check if the command-line arguments are passed
% if exist('a', 'var') == 0 || exist('b', 'var') == 0
%     % Prompt user for inputs
%     a = input('Enter the value for a: ');
%     b = input('Enter the value for b: ');
% end

a=2501;
b=5000;

k = 1;
con = 1;
particles_per_unit = 50 * k;
dx = con / particles_per_unit;
ratio = 1.4;
h = ratio * dx;
c_ = 2 * h;
q = 30;
max_ord = 3;
n_samples = 96 * 50;

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

temp = [X_flat, Y_flat];
coords = zeros(P, J, 2);
coords(1, :, :) = temp;

% Initialize velocity field samples
u_init = zeros(2, J, n_samples);
for i = 1:n_samples
    disp(["Starting the " num2str(i) "th iteration"]);
    u_init(:, :, i) = create_sample(i, particles_per_unit);
end

% Compute covariance matrices for x and y components
u_mean_x = mean(u_init(1, :, :), 3); % Mean of x component
u_mean_y = mean(u_init(2, :, :), 3); % Mean of y component

% Subtract the mean to center the data
u_centered_x = squeeze(u_init(1, :, :) - u_mean_x); % Centered x component
u_centered_y = squeeze(u_init(2, :, :) - u_mean_y); % Centered y component

% Compute covariance matrices
cov_x = (1 / (size(u_centered_x, 2) - 1)) * (u_centered_x * u_centered_x');
cov_y = (1 / (size(u_centered_y, 2) - 1)) * (u_centered_y * u_centered_y');

% Eigen decomposition for x component
[eigenvectors_x, D_x] = eig(cov_x);
eigenvalues_x = diag(D_x);
[eigenvalues_x, idx] = sort(eigenvalues_x, 'descend');
eigenvectors_x = eigenvectors_x(:, idx);

% Eigen decomposition for y component
[eigenvectors_y, D_y] = eig(cov_y);
eigenvalues_y = diag(D_y);
[eigenvalues_y, idx] = sort(eigenvalues_y, 'descend');
eigenvectors_y = eigenvectors_y(:, idx);

% Parallel computation
num_cores = feature('numcores');

parpool(num_cores); % Start parallel pool

start_time = tic;

% Preallocate the results as a struct array instead of a cell
results = zeros(b - a + 1, 2, 2500, 101); % Properly preallocate with correct dimensions

parfor idx = 1:(b - a + 1)
    i = a + idx - 1; % Map idx to 'i'
    results(idx,:,:,:) = reshape(solver_wrapper(i, q, u_mean_x, u_mean_y, eigenvalues_x, eigenvectors_x, eigenvalues_y, eigenvectors_y), [1, 2, 2500, 101]); % Explicitly match dimensions
end

elapsed_time = toc(start_time);

fprintf('Total time taken %.2f mins\n', elapsed_time / 60);


% Convert results (4D array) to float32
results_single = single(results);

% Save the converted results to a file
save('results_mcs2_test2.mat', 'results_single', '-v7.3');

% Function to create a sample
function u0 = create_sample(i, particles_per_unit)
    rng(i);
    L = 4;
    a_ij = randn(2, 2 * L + 1, 2 * L + 1);
    b_ij = randn(2, 2 * L + 1, 2 * L + 1);
    c = rand(2, 1) * 2 - 1;

    % Define the w(x, y) function
    function [result, m] = w(x, y)
        result = zeros(2, size(x, 1), size(x, 2));
        m = ones(2, size(x, 1), size(x, 2));
        for i = -L:L
            for j = -L:L
                result(1, :, :) = squeeze(result(1, :, :)) + (a_ij(1, i + L + 1, j + L + 1) * sin(2 * pi * (i * x + j * y)) + ...
                                b_ij(1, i + L + 1, j + L + 1) * cos(2 * pi * (i * x + j * y)));

                result(2, :, :) = squeeze(result(2, :, :)) + (a_ij(2, i + L + 1, j + L + 1) * sin(2 * pi * (i * x + j * y)) + ...
                                b_ij(2, i + L + 1, j + L + 1) * cos(2 * pi * (i * x + j * y)));
                                
            end
        end
        m(1, :, :) = 10 * (1 - exp(y .* (1 - y))) .* (1 - exp(-x .* (1 - x)));
        m(2, :, :) = 10 * (1 - exp(y .* (1 - y))) .* (1 - exp(-x .* (1 - x)));
    end

    % Calculate w(x, y) over the grid
    [X, Y] = meshgrid(linspace(0, 1, particles_per_unit));
    [W, m] = w(X, Y);

    % Calculate u(x, y, t=0)
    temp = 2 * W / max(abs(W(:)));
    temp(1, :, :) = (temp(1, :, :) + c(1)) .* m(1, :, :);
    temp(2, :, :) = (temp(2, :, :) + c(2)) .* m(2, :, :);
    u0 = zeros(2, numel(X));
    u0(1, :) = reshape(temp(1, :, :), [1, numel(X)]);
    u0(2, :) = reshape(temp(2, :, :), [1, numel(X)]);
end


% Solver wrapper function
function result = solver_wrapper(i, q, u_mean_x, u_mean_y, eigenvalues_x, eigenvectors_x, eigenvalues_y, eigenvectors_y)
    disp(["Starting the " num2str(i) "th iteration"]);
    rng(i);
    wx = randn(1, q);
    wy = randn(1, q);
    result = solver(cal_u(wx, wy, u_mean_x, u_mean_y, eigenvalues_x, eigenvectors_x, eigenvalues_y, eigenvectors_y, q),i);
    disp(["Done with " num2str(i) "th iteration"]);
end

% Function to calculate u(wx, wy)
function val = cal_u(wx, wy, u_mean_x, u_mean_y, eigenvalues_x, eigenvectors_x, eigenvalues_y, eigenvectors_y, q)
    val = zeros(size(u_mean_x));
    val(1, :) = u_mean_x;
    val(2, :) = u_mean_y;
    for i = 1:q
        val(1, :) = val(1, :) + sqrt(eigenvalues_x(i)) * (eigenvectors_x(:, i)' * wx(i));
        val(2, :) = val(2, :) + sqrt(eigenvalues_y(i)) * (eigenvectors_y(:, i)' * wy(i));
    end
end



function ut = solver(u_init, i)
    rng(i); % Seed the random number generator
    
    % Parameters
    k = 1;
    con = 1;
    particles_per_unit = 50 * k;
    dx = con / particles_per_unit;
    ratio = 1.4;
    h = ratio * dx;

    vis = 0.05;
    J = particles_per_unit^2; % Total number of particles
    rho = 1000 * ones(J, 1);  % Density for steel
    mass = rho * dx^2;
    
    T = 0.30;    % Total time of integration
    dt = 0.003;  % Time step
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
    tol = 1e-6;
    x_mask = (abs(coords(:, 1) - 0) < tol) | (abs(coords(:, 1) - 1) < tol);
    y_mask = (abs(coords(:, 2) - 0) < tol) | (abs(coords(:, 2) - 1) < tol);

    mask = (x_mask | y_mask);
   
    % Time-stepping loop
    for n = 2:N+1
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
        rhsx = u1 .* ux_d1 + u2 .* ux_d2 - vis * (ux_ddx + ux_ddy);
        rhsy = u1 .* uy_d1 + u2 .* uy_d2 - vis * (uy_ddx + uy_ddy);

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