function [B, flag] = ModNewton(H, Min, Max)

    % Check if the matrix is sparse
    if issparse(H)
    % Convert sparse matrix to full matrix
    H_full = full(H);
    
    % Compute eigenvalues and eigenvectors using full matrix
    [V, T] = eig(H_full);
    else
    % Compute eigenvalues and eigenvectors directly
    [V, T] = eig(H);
    end


    if all(diag(T) > 0) % IF all the eingenvalues are positive, then no need to modifiy H
        flag = 0;
        B = H;
    
    else
        flag = 1;
        L = abs(T);
        L_diag = diag(L);
        L_diag(L_diag > Max) = Max;
        L_diag(L_diag < Min) = Min;
        L = diag(L_diag); % We modified L using the algorithm of MEthod 2 in class notes
        B = V * L * V';
    end
   
end