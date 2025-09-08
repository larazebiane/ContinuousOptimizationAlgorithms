function [p, info] = cg_tr(Hv_hand, g, radius, params)
%CG_TR Conjugate Gradient Method with Trust Region Constraint
%
%   [p, info] = cg_tr(Hv_hand, g, radius, params)
%
%   Solves the trust-region subproblem approximately using Conjugate Gradient:
%       minimize 0.5 * p' * H * p + g' * p, subject to ||p|| <= radius
%
%   INPUTS:
%     Hv_hand : Function handle to compute Hessian-vector product Hv_hand(v)
%     g       : Gradient vector at current point
%     radius  : Trust-region radius (positive scalar)
%     params  : Struct with optional parameters:
%               - tol        : tolerance for residual norm (default 1e-6)
%               - maxiter    : max number of iterations (default 500)
%               - printlevel : verbosity level (default 1)
%
%   OUTPUTS:
%     p    : Computed step within trust region
%     info : Struct with fields:
%            - status : 0 if converged, 1 if maxiter reached,
%                       -1 or 2 if boundary hit
%            - res    : final residual norm
%            - iter   : number of iterations performed

    % Set default parameters if not provided
    default_params.tol = 1e-6;
    default_params.maxiter = 500;
    default_params.printlevel = 1;

    if nargin < 4
        params = default_params;
    else
        % Fill in any missing params with defaults
        param_fields = fieldnames(default_params);
        for i = 1:length(param_fields)
            if ~isfield(params, param_fields{i})
                params.(param_fields{i}) = default_params.(param_fields{i});
            end
        end
    end

    % Initialization
    p = zeros(size(g));   % Current estimate of solution vector
    r = g;                % Residual vector (initially gradient)
    s = -g;               % Search direction (steepest descent)
    iter = 0;

    % Main CG loop
    while true
        % Check convergence: residual norm small enough?
        if norm(r) <= params.tol * max(1, norm(g))
            info.status = 0; % Converged successfully
            break;
        elseif iter >= params.maxiter
            info.status = 1; % Max iterations reached without convergence
            break;
        end

        % Compute Hessian-vector product for direction s
        Bs = Hv_hand(s);
        sBs = s' * Bs;

        if sBs > 0
            % Calculate step length along conjugate direction
            alpha = (r' * r) / sBs;
        else
            % Negative curvature or non-positive curvature detected
            % Find tau so that ||p + tau * s|| = radius
            coeffs = [s' * s, 2 * (s' * p), p' * p - radius^2];
            tau_candidates = roots(coeffs);
            tau = max(tau_candidates(imag(tau_candidates) == 0 & tau_candidates > 0));
            p = p + tau * s;

            info.status = -1; % Boundary point reached due to curvature
            break;
        end

        % Check if next step stays inside trust region
        if norm(p + alpha * s) < radius
            % Update step and residual
            p = p + alpha * s;
        else
            % Step crosses trust region boundary
            % Compute tau to exactly hit boundary
            coeffs = [s' * s, 2 * (s' * p), p' * p - radius^2];
            tau_candidates = roots(coeffs);
            tau = max(tau_candidates(imag(tau_candidates) == 0 & tau_candidates > 0));
            p = p + tau * s;

            info.status = 2; % Boundary point reached
            break;
        end

        % Update residual and conjugate direction
        r_prev = r;
        r = r + alpha * Bs;
        beta = (r' * r) / (r_prev' * r_prev);
        s = -r + beta * s;

        iter = iter + 1;
    end

    % Prepare output info
    info.res = norm(r);
    info.iter = iter;
    info.p = p;

    % Print iteration summary if requested
    if params.printlevel > 0
        fprintf('Iteration: %d\n', iter);
        fprintf('Residual norm: %e\n', info.res);
        fprintf('---------------------------------------------\n');
    end

end
