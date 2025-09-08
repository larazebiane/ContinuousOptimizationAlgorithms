function [x, info] = bqp_pgT(c, H, l, u, x0, params)
    % Initialize variables
    maxit = params.maxit;
    printlevel = params.printlevel;
    tol = params.tol;
    eta = 0.1; 
    status=0;
   

    % Initialize the starting point and compute the initial objective value, gradient value and projection function value
    x = x0;
    q = @(x) 0.5 * x' * H * x + c' * x;
    grad_q = @(x) H * x + c;
    
    % Initialize iteration counter and info structure
    iter = 0;
    
    
    P = @(x)   min(max(x,l),u);  %projection function
    x=P(x);






while true

        n=length(x);
        zl    = zeros(n,1);
        zu    = zeros(n,1);


        grad = grad_q(x);
        obj = q(x);
        for i = 1 : n
            if x(i) == l(i)
                zl(i) = grad(i);
            elseif x(i) == u(i)
                zu(i) = -grad(i);
            end
        end
        
        res = grad - zl + zu;

        % Store value of initial KKT residual
        if iter == 0
            norm_0 = norm(res);
        end

        if printlevel > 0
        fprintf('Iter %d, obj = %.4f, ||r|| = %7.4e+\n', iter, obj, norm(res));
        end

        
        if iter > maxit
            info.status = 1;
            break;
        end
        
    

        % Check the condition of non-negative dual variables and convergence
        if all([zl;zu] >= 0) && (norm(res) < tol * norm_0)
            info.status = 0; % KKT condition satisfied
            fprintf('\nbqp_pg: KKT condition satisfied, algorithm exit.\n');
            break
        end


        % Line search to find the acceptable step size
        
        alpha = 1;
        d = P(x - alpha * grad) - x;
        while q(x + d) > q(x) + eta * grad' * d
            alpha = 0.5 * alpha;
            d = P(x - alpha * grad) - x;
        end
        % Return Inexact Cauchy point
        x = x + d;



        
        
        if params.ssm == 1
           
            
            xc=x;
            
            % Determine the active set AE(xc)
            

            m = length(xc);  % Length of xc
            AE = zeros(m, 1);  % Initialize active set vector
            xl = [];  % Initialize vector for boundary values
            xi = [];  % Initialize vector for non-boundary values
            
            % Loop to populate AE, xl, and xi
            for i = 1:m
                if xc(i) == l(i) || xc(i) == u(i)
                    AE(i) = 1;  % Mark as active
                    xl = [xl; xc(i)];  % Append boundary value to xl
                else
                    AE(i) = 0;  % Mark as not active
                    xi = [xi; xc(i)];  % Append non-boundary value to xi
                end
            end
            
            % Determine indices for non-active set variables
            freeIndices = find(AE == 0);  % Indices where AE(i) is 0

            % Not free indices where AE is 1
            notFreeIndices = AE == 1;
            
            % Create the submatrix HII containing only elements from non-active indices
            HII = H(freeIndices, freeIndices);
            H_cross = H(freeIndices, notFreeIndices);
                                          
           
            cf = c(freeIndices);
            
           

            % Solve the reduced system (assuming Hff is non-singular) using Newton's
            xf = -HII \ (cf+H_cross*xl);  % Solves Hff * xf = -cf for xf



        
            % Construct the full solution xs
            xs = xc;  % Start with xc
            xs(~AE) = xf;  % Update free variables with optimized values

            direction = xs - xc; % Compute the direction vector from xc to xs
            maxIter = 100;  % Limit on iterations to prevent infinite loops

            % Update rule based on the gradient condition
            if grad_q(xc)' * direction < 0
                % Initialize parameters for line search
                j = 0;      % Initial exponent for (1/2)^j
            
                % Backtracking line search to find the smallest j
                while true
                    newPoint = P(xc + (1/2)^j * direction);  % Apply projection function
                    if q(newPoint) <= q(xc)
                        break;  % Condition met, exit loop
                    else
                        j = j + 1;  % Increase j to increase the exponent of 1/2
                        if j > maxIter  % Safety check
                            error('Maximum iterations exceeded during line search');
                        end
                    end
                end
            
                x = newPoint;  % Use the new point from the line search
            else
                x = xc;  % Retain the current point
            end


        end

        iter = iter + 1;

       
        info.obj = obj;
        info.iter = iter;
        info.status=status;
        info.res = res;  % Residual as the norm of the gradient
        info.zl = zl;  % Dual variable for lower bound
        info.zu = zu;  % Dual variable for upper bound
        
end