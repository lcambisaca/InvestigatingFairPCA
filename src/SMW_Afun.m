function U = SMW_Afun(b,M,F,G) 
    y1 = (M*F)\(M*b); 
    y2 = b - F*y1; 
    U = G*y2;
end


