function [p, info] = cg_ls(Hv_hand, g, params)


default_params.tol = 1e-8;
default_params.maxiter = 500;
default_params.printlevel = 1;  % No printing by default

% Check number of input arguments
if nargin < 3
  % Use default parameters if optional arguments are missing
  params = default_params;
end 

% Initialize variables
p = zeros(size(g));
r = g;
s = -g;
k = 0;
info.status=0;

% Loop until convergence or maximum iterations reached
while norm(r) > params.tol*max(1,norm(g)) && k < params.maxiter
  % Check for negative curvature
  Hvs=Hv_hand(s); % Used so that we don't have to calculate Hv_hand(s) each time
  if dot(s, Hvs) > 0
    % Update step length
    alpha = dot(r, r) / dot(s, Hvs);
  else
      info.status = -1;
      
    % Negative curvature encountered
    if k == 0
      % Special case: return negative gradient if k=0
      p = -g;
      return;
    end
    break;
  end
  
  % Update iterate and residual

  rprev = r;
  p = p + alpha * s;
  r = r + alpha * Hvs;
  
  % Update conjugate direction
  beta = dot(r, r) / dot(rprev, rprev);
  s = -r + beta * s;
  
  % Print information if required
  if params.printlevel > 0
    fprintf('Iteration: %d, Residual: %e\n', k+1, norm(r));
  end
  
  % Update counters
  k = k + 1;
end

% Set output parameters
info.res = norm(r);
info.iter = k;
if k == params.maxiter
  info.status = 1;
end

end
