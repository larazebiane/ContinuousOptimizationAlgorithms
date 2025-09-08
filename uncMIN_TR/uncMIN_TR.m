function [x, info] = uncMIN_TR(fun_hands, x0, params)
%UNCMIN_TR Trust Region Method for Unconstrained Minimization
%
%   [x, info] = uncMIN_TR(fun_hands, x0, params)
%
%   Solves: minimize f(x), using Trust Region strategies.
%
%   INPUTS:
%     fun_hands: Struct with function handles
%       - f_hand(x) : Objective function
%       - g_hand(x) : Gradient
%       - H_hand(x) : Hessian
%       - Hv_hand(x, v) : Hessian-vector product
%
%     x0       : Initial point
%     params   : Struct with fields:
%       - step_type   : 'CauchyStep', 'NewtonCG', or 'More-Sorensen'
%       - maxit       : Maximum iterations
%       - tol         : Gradient norm stopping tolerance
%       - printlevel  : Verbosity level
%
%   OUTPUTS:
%     x    : Final solution
%     info : Struct with detailed output (iter count, f_values, etc.)

    % === Extract parameters ===
    step_type  = params.step_type;
    maxit      = params.maxit;
    tol        = params.tol;
    printlevel = params.printlevel;

    % === Initialize variables ===
    x = x0;
    delta = 1;  % Initial trust region radius
    iter = 0;
    f_evals = 0;
    g_evals = 0;
    H_evals = 0;
    Hv_evals = 0;
    
    % Initialize info struct fields
    info = struct('f', [], 'g', [], 'H', [], 'Hv', [], ...
                  'g_norm', 0, 'status', 0, 'f_values', []);

    % === Extract function handles ===
    f_hand  = fun_hands.f_hand;
    g_hand  = fun_hands.g_hand;
    H_hand  = fun_hands.H_hand;
    Hv_hand = fun_hands.Hv_hand;

    % === Initial evaluation ===
    norm_x  = norm(x);
    f       = feval(f_hand, x);
    g       = feval(g_hand, x);
    norm_g  = norm(g);
    norm_g0 = norm_g;
    H       = feval(H_hand, x);
    
    % Store function values
    f_values = f;
    info.f_values = f_values;

    % === Main optimization loop ===
    while true
        g_evals = g_evals + 1;

        % === Stopping Criteria ===
        if norm(g) <= tol * max(1, norm_g0)
            status = 0;  % Converged
            break;
        elseif iter >= maxit
            status = 1;  % Max iterations reached
            break;
        end

        % === Compute Search Direction Based on Step Type ===
        if strcmp(step_type, 'NewtonCG')
            Hv_hand_mod = @(s) Hv_hand(x, s);
            s = cg_tr(Hv_hand_mod, g, delta);
            x = x + s;
            Hv_evals = Hv_evals + 1;
            f_evals  = f_evals + 1;

        elseif strcmp(step_type, 'More-Sorensen')
            s = more_sorensen(H, g, 0.01, params);
            x = x + s;
            H_evals = H_evals + 1;
            f_evals = f_evals + 1;

        elseif strcmp(step_type, 'CauchyStep')
            % Build quadratic model: m_k(s) = -gᵗs - 0.5*sᵗHs
            delm_k = @(s) - g' * s - 0.5 * s' * H * s;

            % Compute Hessian-vector product (gᵗHg)
            Bg   = feval(Hv_hand, x, g);
            gBg  = g' * Bg;
            Hv_evals = Hv_evals + 1;

            % Compute step length alpha
            if gBg <= 0
                alpha = delta / norm(g);
            elseif delta / norm(g) <= (norm(g)^2) / gBg
                alpha = delta / norm(g);
            else
                alpha = (norm(g))^2 / gBg;
            end

            % Take step
            s = -alpha * g;

            % Evaluate actual reduction
            f_new = feval(f_hand, x + s);
            rho   = (f - f_new) / delm_k(s);
            f_evals = f_evals + 1;

            % Update x and trust region radius based on rho
            if rho >= 0.9
                x = x + s;
                delta = 2 * delta;
            elseif rho >= 0.1
                x = x + s;
                % keep delta unchanged
            else
                % step rejected
                delta = 0.5 * delta;
            end
        else
            error('Invalid step_type. Allowed values: CauchyStep, NewtonCG, More-Sorensen.');
        end

        % === Update iteration state ===
        iter = iter + 1;
        norm_x = norm(x);
        f = feval(f_hand, x);
        g = feval(g_hand, x);
        norm_g = norm(g);
        f_values(iter+1) = f;
    end

    % === Final printing ===
    if printlevel
        if status == 0
            fprintf('Relative stopping tolerance reached.\n');
        else
            fprintf('Maximum allowed iterations reached.\n');
        end
        fprintf('     ||x||    : %13.7e\n', norm_x);
        fprintf(' Iterations   : %-5g\n', iter);
        fprintf(' Final f      : %13.7e\n', f );
        fprintf(' Final ||g||  : %13.7e\n', norm_g );
        fprintf(' Final ||g0|| : %13.7e\n', norm_g0 );
    end

    % === Output info struct ===
    info.x = x;
    info.f = f;
    info.g = g;
    info.f_evals = f_evals;
    info.g_evals = g_evals;
    info.norm_g = norm_g;
    info.H = H;
    info.H_evals = H_evals;
    info.Hv = Hv;
    info.Hv_evals = Hv_evals;
    info.iter = iter;
    info.status = status;
    info.f_values = f_values;

end
