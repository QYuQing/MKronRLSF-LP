function [A] = kronrls_lap(k1,k2,y, sita)
    d1 = diag(sum(k1,1).^(-0.5));
    d2 = diag(sum(k2,1).^(-0.5));
    w1 = d1*k1*d1;
    w2 = d2*k2*d2;
    
    [Qa,la] = eig(w1);
    [Qb,lb] = eig(w2);
    ll = kron(diag(lb)',diag(la));
    inverse = 1 ./ (1+sita*(1-ll));
%     inverse = 1 ./ (1 + lambda*(1 - ll));
    m1 = Qa' * y * Qb;
    m2 = inverse.*m1;
    A = Qa * m2 * Qb';      
end