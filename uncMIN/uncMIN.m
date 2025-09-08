function [x, info] = uncMIN(fun_hands, x0, params)
%UNCMin General unconstrained minimization routine
%
%   [x, info] = uncMIN(fun_hands, x0, params)
%
%   Solves:    minimize f(x)
%
%   INPUTS:
%     fun_hands : Struct containing function handles:
%                   * f_hand  : objective function
%                   * g_hand  : gradient function
%                   * H_hand  : full Hessian function
%                   * Hv_hand : Hessian-vector product function
%     x0        : Initial guess
%     params    : Struct with fields:
%                   * dir_type    : 'SteepestDescent', 'ModifiedNewton', or 'NewtonCG'
%                   * maxit       : Maximum number of iterations
%                   * tol         : Convergence tolerance
%                   * printlevel  : 0 = silent, 1 = iteration output
%
%   OUTPUTS:
%     x     : Final solution vector
%     info  : Struct with detailed outputs:
%               * f, g        : Final objective and gradient
%               * f_evals     : Number of objective evaluations
%               * g_evals     : Number of gradient evaluations
%               * H_evals     : Number of Hessian evaluations
%               * Hv_evals    : Number of Hv-products (for NewtonCG)
%               * iter        : Number of iterations
%               * status      : 0 = converged, 1 = max iterations reached
%               * f_values    : History of function values
%               * sta         : Status from direction solver (e.g., NewtonCG)
%

    % Initialize
    x = x0;
    iter = 0;
    f_evals = 0;
    g_evals = 0;
    H_evals = 0;
    Hv_evals = 0;
    status = 0;

    % Extract function handles
    f_hand = fun_hands.f_hand;
    g_hand = fun_hands.g_hand;
    H_hand = fun_hands.H_hand;
    Hv_hand = fun_hands.Hv_hand;

    % Evaluate initial function, gradient, and Hessian
    norm_x  = norm(x);
    f       = feval(f_hand, x);
    g       = feval(g_hand, x);
    H       = feval(H_hand, x);
    norm_g  = norm(g);
    norm_g0 = norm_g;

    % Extract parameters
    dir_type    = params.dir_type;
    maxit       = params.maxit;
    tol         = params.tol;
    printlevel  = params.printlevel;

    % Store objective value history
    f_values = f;
    
    % Main optimization loop
    while true
        if printlevel > 0
            fprintf('Iteration: %d, Function value: %.6e, Gradient norm: %.2e\n', iter, f, norm(g));
        end

        g_evals = g_evals + 1;

        % Check stopping criteria
        if norm(g) <= tol * max(1, norm_g0)
            status = 0; break;
        elseif iter >= maxit
            status = 1; break;
        end

        % Compute search direction
        switch dir_type
            case 'SteepestDescent'
                p = -g;

            case 'ModifiedNewton'
                [B, sta] = ModNewton(H, 1e-6, 1e6);
                H_evals = H_evals + 1;
                p = B \ (-g);

            case 'NewtonCG'
                [p, sta] = cg_ls(@(s) feval(Hv_hand, x, s), g);
                Hv_evals = Hv_evals + 1;

            otherwise
                error('Invalid dir_type. Allowed: SteepestDescent, ModifiedNewton, NewtonCG.');
        end

        % Perform line search
        alpha = line_search(x, p, f_hand, g_hand, 1, 0.1, 0.9);
        f_evals = f_evals + 1;

        % Update iterate
        x = x + alpha * p;
        iter = iter + 1;

        % Evaluate new point
        norm_x = norm(x);
        f      = feval(f_hand, x);
        g      = feval(g_hand, x);
        norm_g = norm(g);
        f_values(iter + 1) = f;  % Store objective value
    end

    % Final summary
    if printlevel
        if status == 0
            fprintf('Converged: relative gradient tolerance reached.\n');
        else
            fprintf('Terminated: maximum iterations reached.\n');
        end
        fprintf('  ||x||     : %13.7e\n', norm_x);
        fprintf('  Iterations: %-5d\n', iter);
        fprintf('  Final f   : %13.7e\n', f);
        fprintf('  Final ||g||  : %13.7e\n', norm_g);
        fprintf('  Initial ||g0||: %13.7e\n', norm_g0);
    end

    % Pack outputs
    info.x         = x;
    info.f         = f;
    info.g         = g;
    info.f_evals   = f_evals;
    info.g_evals   = g_evals;
    info.norm_g    = norm_g;
    info.H         = H;
    info.H_evals   = H_evals;
    info.Hv        = [];           % Not used directly
    info.Hv_evals  = Hv_evals;
    info.iter      = iter;
    info.status    = status;
    info.f_values  = f_values;
    info.sta       = sta;
end
