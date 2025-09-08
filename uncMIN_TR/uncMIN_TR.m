function [x, info] = uncMIN_TR(fun_hands, x0, params)
    
    % Extract parameters from params structure
    step_type = params.step_type;
    maxit = params.maxit;
    printlevel = params.printlevel;
    tol = params.tol;
    
    % Initialize variables
    x = x0;
    delta=1;
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
    

    % Extract function handles from fun_hands structure
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

    % Initialize f_values to store the objective function value at each iteration
    f_values = f;  % Preallocate f_values
    info.f_values = f_values;

    % Main loop
    while (1)
        
        g_evals = g_evals + 1; % Increment g_evals by 1 as we use it in the next step
        % Check for termination
        if norm(g) <= tol*max(1,norm_g0)
            status = 0;
            break;
        elseif (iter >= maxit)
            status = 1;
            break
        end
        
        if strcmp(step_type, 'NewtonCG')
            Hv_hand_modified = @(s) Hv_hand(x, s);
            s=cg_tr(Hv_hand_modified,g,1);
            x=x+s;
            Hv_evals=Hv_evals+1;
            f_evals=f_evals+1;

        elseif strcmp(step_type, 'More-Sorensen')
                s=more_sorensen(H,g,0.01,params);
                x=x+s;
                H_evals=H_evals+1;
                f_evals=f_evals+1;
        elseif strcmp(step_type, 'CauchyStep')
              
            % Build second-order model
            delm_k = @(s) - g' * s - 0.5 * s' * H * s;
            
            % Calculate g'Bg
            Bg=feval(Hv_hand, x, g);
            gBg=g'*Bg;
            Hv_evals=Hv_evals+1; %Update Hv_evals
            
            % Solve trust-region subproblem
            if gBg <= 0
                alpha = delta / norm(g);
            elseif delta / norm(g) <= (norm(g)^2) / (gBg)
                alpha = delta / norm(g);
            else
                alpha= (norm(g))^2/(gBg);
            end
            
            % Compute actual reduction
            s=-alpha*g;



            f_new = feval(f_hand, x + s);
            rho = (f - f_new) / (delm_k(s));
            
            f_evals=f_evals+1; % Update f_evals as we evaluate f in the previous step
            
            % Update variables based on actual reduction
            if rho >= 0.9
                x = x + s;
                delta = 2 * delta;
            elseif rho >= 0.1
                x = x + s;
                %delta = delta;
            else
                %x = x;
                delta = 0.5 * delta;
            end


        else
            error('Invalid step_type. Allowed value are CauchyStep, NewtonCG, or More-Sorensen only.');
        end

        
        % Update iteration counter
        iter = iter + 1;
        norm_x = norm(x);
        f      = feval(f_hand,x);
        g      = feval(g_hand,x);
        norm_g = norm(g);
        % Store the objective function value at this iteration
        f_values(iter+1) = f; 
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
    info.f_values = f_values;

end
