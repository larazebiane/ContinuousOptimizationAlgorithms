function [p, info] = cg_tr(Hv_hand, g, radius, params)
    % Inputs:
    % Hv_hand: Function handle for computing Hv for a given vector v
    % g: Vector with length equal to the number of optimization variables
    % radius: Trust-region radius
    % params: MATLAB structure containing parameters (tol, maxiter, printlevel)
    
    % Outputs:
    % s: Computed direction
    % info: MATLAB structure containing status, residual, and number of iterations
    
    % Initialize parameters
   

    default_params.tol = 1e-6;
    default_params.maxiter = 500;
    default_params.printlevel = 1;

    if nargin < 4
  % Use default parameters if optional arguments are missing
  params = default_params;
    end 
    
    % Initialize variables
    p = zeros(size(g));
    r = g;
    s = -g;
    iter = 0;
    
    % Main loop
    while (1)
        
        % Check for termination
        if norm(r) <= params.tol*max(1,g)
            info.status = 0;
            break;
        elseif (iter >= params.maxiter)
            info.status = 1;
            break
        end
        
        Bs=Hv_hand(s);
        sBs=s' * Hv_hand(s);

        if sBs > 0
            alpha = (r' * r) / sBs;
        else
            % Check if ||p + τs||^2 < radius^2
            tau = roots([s' * s, s'*p+ p' * s, p' * p - radius^2]);
            tau = max(tau(real(tau) > 0));
            p = p + tau * s;

            info.status = -1; % Boundary point
            return;
        end
        
        if norm(p + alpha * s) < radius
            p = p + alpha * s;
        else
            % Check if ||p + τs||^2 < radius^2
            
            tau = roots([s' * s, s'*p+ p' * s, p' * p - radius^2]);
            tau = max(tau(real(tau) > 0));
            p = p + tau * s;


            info.status = 2; % Boundary point
            return;
        end
        

        rPRE=r;
        r = r + alpha * Bs;
        beta = (r' * r) / (rPRE' * rPRE);
        s = -r + beta * s;
        iter = iter + 1;
    end
    
    
    % Finalize outputs
    info.res = norm(r);
    info.iter = iter;
    info.p=p;

        if params.printlevel > 0
                fprintf('Iteration: %d\n', iter);
                fprintf('norm(r) = %f\n', norm(r));
                fprintf('---------------------------------------------\n');
        else
            fprintf('no');
        end
end
