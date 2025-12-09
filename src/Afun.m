function Apb = Afun(A, C, b, sigma)
    
    y1 = (C'*C)\(C'*b);
%     Cy1nnz = nnz(C*y1)/size(C,1)
    y2 = b - C*y1;
    % A*P*b
    
    y3 = A*y2;
    
%     [y4, flag2] = lsqr(C, y3);
%     y4 = C\y3;
    y4 = (C'*C)\(C'*y3);
    Apb = y3 - C*y4 - sigma*y2 + sigma*b;
    
end

