function [B, flag] = ModNewton(H, Min, Max)
%MODNEWTON  Regularizes the Hessian matrix for Modified Newton method
%
%   [B, flag] = ModNewton(H, Min, Max)
%
%   This function modifies the Hessian matrix H to ensure it is
%   positive definite by bounding its eigenvalues within [Min, Max].
%
%   INPUTS:
%     H   : Hessian matrix (n x n)
%     Min : Minimum allowed eigenvalue
%     Max : Maximum allowed eigenvalue
%
%   OUTPUTS:
%     B    : Modified Hessian matrix with bounded eigenvalues
%     flag : 0 if no modification was needed, 1 if modification applied
%

    % If H is sparse, convert to full matrix for eigendecomposition
    if issparse(H)
        H_full = full(H);
        [V, T] = eig(H_full);
    else
        [V, T] = eig(H);
    end

    % Check if all eigenvalues are positive
    if all(diag(T) > 0)
        % H is already positive definite — no modification needed
        flag = 0;
        B = H;
    else
        % Modify eigenvalues to ensure positive definiteness
        flag = 1;

        % Take absolute values of eigenvalues
        L = abs(T);
        L_diag = diag(L);

        % Clamp eigenvalues to [Min, Max]
        L_diag(L_diag > Max) = Max;
        L_diag(L_diag < Min) = Min;

        % Reconstruct regularized matrix
        L = diag(L_diag);
        B = V * L * V';
    end
end
