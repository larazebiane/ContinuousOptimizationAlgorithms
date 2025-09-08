function [p, info] = cg_ls(Hv_hand, g, params)
%CG_LS Truncated Conjugate Gradient Method for Newton systems
%
%   [p, info] = cg_ls(Hv_hand, g, params)
%
%   Solves approximately: H * p = -g using truncated Conjugate Gradient.
%
%   INPUTS:
%     Hv_hand   - Function handle to compute Hessian-vector product: Hv = Hv_hand(v)
%     g         - Gradient vector
%     params    - Struct with fields:
%                   * tol        : Convergence tolerance (default 1e-8)
%                   * maxiter    : Maximum iterations (default 500)
%                   * printlevel : 0 = silent, 1 = show residuals (default 1)
%
%   OUTPUTS:
%     p         - Computed search direction
%     info      - Struct with fields:
%                   * res    : Final residual norm
%                   * iter   : Number of CG iterations
%                   * status : -1 = negative curvature,
%                               0 = converged,
%                               1 = max iterations reached

    % Set default parameters
    default_params.tol = 1e-8;
    default_params.maxiter = 500;
    default_params.printlevel = 1;

    % Use default params if not provided
    if nargin < 3
        params = default_params;
    end

    % Initialize variables
    p = zeros(size(g));
    r = g;
    s = -g;
    k = 0;
    info.status = 0;

    % Main CG loop
    while norm(r) > params.tol * max(1, norm(g)) && k < params.maxiter
        Hvs = Hv_hand(s);  % Hessian-vector product

        % Check for negative curvature
        if dot(s, Hvs) > 0
            % Compute step length
            alpha = dot(r, r) / dot(s, Hvs);
        else
            info.status = -1;  % Negative curvature
            if k == 0
                % Return negative gradient if first iteration
                p = -g;
                return;
            end
            break;
        end

        % Update iterate and residual
        r_prev = r;
        p = p + alpha * s;
        r = r + alpha * Hvs;

        % Compute new conjugate direction
        beta = dot(r, r) / dot(r_prev, r_prev);
        s = -r + beta * s;

        % Print info if enabled
        if params.printlevel > 0
            fprintf('  CG Iteration: %d, Residual: %.2e\n', k + 1, norm(r));
        end

        % Increment iteration counter
        k = k + 1;
    end

    % Final outputs
    info.res = norm(r);
    info.iter = k;
    if k == params.maxiter
        info.status = 1;  % Max iterations reached
    end
end
