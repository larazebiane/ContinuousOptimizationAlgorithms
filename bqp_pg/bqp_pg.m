function [x, info] = bqp_pg(c, H, l, u, x0, params)

    % bqp_pg - Projected Gradient Method for Bound-Constrained Quadratic Optimization
    %
    % Solves:
    %   minimize    q(x) = cᵗx + ½ xᵗHx
    %   subject to  l ≤ x ≤ u
    %
    % Inputs:
    %   c       - Linear term (n x 1)
    %   H       - Hessian matrix (n x n), symmetric positive definite
    %   l, u    - Lower and upper bounds (n x 1)
    %   x0      - Initial point (n x 1)
    %   params  - Struct with fields:
    %               .maxit       - Max iterations
    %               .tol         - Relative KKT tolerance
    %               .ssm         - Use subspace acceleration (0 or 1)
    %               .printlevel  - 0 (silent) or 1 (verbose)
    %
    % Outputs:
    %   x       - Final iterate
    %   info    - Struct with:
    %               .obj     - Final objective value
    %               .res     - Final KKT residual
    %               .iter    - Number of iterations
    %               .zl, .zu - Dual variables for bounds
    %               .status  - 0 (converged), 1 (max iter)
    
    % ------------------------ Initialization ------------------------
    maxit      = params.maxit;
    printlevel = params.printlevel;
    tol        = params.tol;
    eta        = 0.1;
    
    x = min(max(x0, l), u); % Projected initial point
    n = length(x);
    iter = 0;
    status = 0;
    
    % Function handles
    q       = @(x) 0.5 * x' * H * x + c' * x;
    grad_q  = @(x) H * x + c;
    project = @(x) min(max(x, l), u);
    
    % ------------------------ Main Loop ------------------------
    while true
        grad = grad_q(x);
        obj  = q(x);
    
        % Dual variables (KKT multipliers)
        zl = zeros(n,1); zu = zeros(n,1);
        for i = 1:n
            if x(i) == l(i)
                zl(i) = grad(i);
            elseif x(i) == u(i)
                zu(i) = -grad(i);
            end
        end
    
        res = grad - zl + zu;
        if iter == 0
            norm0 = norm(res);
        end
    
        % Verbose output
        if printlevel > 0
            fprintf('Iter %d:  obj = %.6f   ||res|| = %.2e\n', iter, obj, norm(res));
        end
    
        % Convergence check
        if norm(res) < tol * norm0 && all([zl; zu] >= 0)
            status = 0;
            if printlevel > 0
                fprintf('Converged: KKT conditions satisfied.\n');
            end
            break;
        end
    
        if iter >= maxit
            status = 1;
            if printlevel > 0
                fprintf('Stopped: Max iterations reached.\n');
            end
            break;
        end
    
        % ------------------ Inexact Cauchy Step ------------------
        alpha = 1;
        d = project(x - alpha * grad) - x;
        while q(x + d) > q(x) + eta * grad' * d
            alpha = 0.5 * alpha;
            d = project(x - alpha * grad) - x;
        end
        xc = x + d;
    
        % ------------------ Subspace Acceleration ------------------
        if params.ssm == 1
            AE = (xc == l) | (xc == u);
            freeIdx = find(~AE);
    
            if ~isempty(freeIdx)
                H_ff     = H(freeIdx, freeIdx);
                H_cross  = H(freeIdx, AE);
                c_f      = c(freeIdx);
                x_active = xc(AE);
    
                % Solve reduced system
                xf = -H_ff \ (c_f + H_cross * x_active);
    
                xs = xc;
                xs(freeIdx) = xf;
    
                direction = xs - xc;
    
                % Backtracking line search
                if grad_q(xc)' * direction < 0
                    j = 0;
                    while true
                        newPoint = project(xc + (1/2)^j * direction);
                        if q(newPoint) <= q(xc) || j > 20
                            break;
                        end
                        j = j + 1;
                    end
                    x = newPoint;
                else
                    x = xc;
                end
            else
                x = xc;
            end
        else
            x = xc;
        end
    
        iter = iter + 1;
    end
    
    % ------------------------ Output ------------------------
    info.obj    = q(x);
    info.res    = res;
    info.iter   = iter;
    info.status = status;
    info.zl     = zl;
    info.zu     = zu;

end
