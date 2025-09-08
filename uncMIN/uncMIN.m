function [x, info] = uncMIN(fun_hands, x0, params) %The function takes three inputs: f un hands, x0, and params
    
    % Initialize and set dummy values for output variables and counters
    x = x0;
    iter = 0;
    f_evals = 0;
    g_evals = 0;
    H_evals = 0;
    Hv_evals = 0;
    f      = [];  info.f = f;
    g      = [];  info.g = g;
    H      = []; info.H=H;
    Hv      = []; info.Hv=Hv;
    g_norm = 0 ;  info.g_norm = g_norm;
    status = 0 ;  info.status = status;
    

    
    % Extract function handles
    f_hand = fun_hands.f_hand;
    g_hand = fun_hands.g_hand;
    H_hand = fun_hands.H_hand;
    Hv_hand = fun_hands.Hv_hand;
    
    % Initialization
    norm_x  = norm(x);
    f       = feval(f_hand,x);
    g       = feval(g_hand,x);
    norm_g  = norm(g);
    norm_g0 = norm_g;
    H = feval(H_hand, x); 
    
    % Extract parameters
    dir_type = params.dir_type;
    maxit = params.maxit;
    tol = params.tol;
    printlevel = params.printlevel;
    
    % Initialize f_values to store the objective function value at each iteration
    f_values = f;  % Preallocate f_values
    info.f_values = f_values;
    % Main loop
    while (1)
        
        % Print information if printlevel is not zero
        if printlevel ~= 0
            fprintf('Iteration: %d, Function value: %f, Gradient norm: %f\n', iter, f, norm(g));
        end
        
        g_evals = g_evals + 1; % Increment g_evals by 1 as we use it in the next step
        % Check for termination
        if norm(g) <= tol*max(1,norm_g0)
            status = 0;
            break;
        elseif (iter >= maxit)
            status = 1;
            break
        end
        
        
        % Compute search direction based on dir_type
        if strcmp(dir_type, 'SteepestDescent')
            p = -g;
        elseif strcmp(dir_type, 'ModifiedNewton')
            [B, sta] = ModNewton(H, 1e-6, 1e6);
            H_evals = H_evals + 1; % Increment H_evals by 1 as we use this evaluation to find the eigenvalues
            % Solve for p
            p = B \ (-g);
        elseif strcmp(dir_type, 'NewtonCG')
            [p,sta] = cg_ls(@(s) feval(Hv_hand, x, s), g);
            Hv_evals=Hv_evals+1;
        else
            error('Invalid dir_type. Allowed values are SteepestDescent and ModifiedNewton.');
        end
        % Perform line search
        alpha = line_search(x, p, f_hand, g_hand, 1, 0.1, 0.9);
        f_evals = f_evals + 1;
        % Update x
        x = x + alpha * p;
        % Update iteration counter
        iter = iter + 1;
        norm_x = norm(x);
        f      = feval(f_hand,x);
        g      = feval(g_hand,x);
        norm_g = norm(g);
        % Store the objective function value at this iteration
        f_values(iter+1) = f;  % Store the value at the current iteration
    end
    % Trim f_values to the actual number of iterations
    f_values = f_values(1:iter);
    
 if printlevel
  if status == 0
    fprintf(' Relative stopping tolerance reached');
  else
      fprintf(' Maximum allowed iterations reached');
  end
  fprintf('     ||x||    : %13.7e\n', norm_x);
  fprintf(' Iterations   : %-5g\n', iter);
  fprintf(' Final f      : %13.7e\n', f );
  fprintf(' Final ||g||  : %13.7e\n', norm_g );
  fprintf(' Final ||g0|| : %13.7e\n', norm_g0 );
 end
 
    % Update info structure
    info.x = x;
    info.f = f;
    info.f_evals = f_evals;
    info.g = g;
    info.g_evals = g_evals;
    info.norm_g = norm(g);
    info.H = H;
    info.H_evals = H_evals;
    info.Hv = Hv;
    info.Hv_evals = Hv_evals;
    info.iter = iter;
    info.status = status;
    info.f_values = f_values;  % Add this line to store the objective function values
    info.sta=sta;
end
