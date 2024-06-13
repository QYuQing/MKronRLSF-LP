function [A,value] = kronrls(k1,k2,y, lambda)
    if ~exist('lambda','var') || isempty(lambda)
        lambda = 1;
    end
    [Qa,la] = eig(k1);
    [Qb,lb] = eig(k2);
    ll = kron(diag(lb)',diag(la));
    inverse = ll ./ (ll + lambda);
%     inverse = 1 ./ (1 + lambda*(1 - ll));
    m1 = Qa' * y * Qb;
    m2 = m1 .* inverse;
    A = Qa * m2 * Qb';   
    
    %计算目标函数值
%     D = (ll.^3)./((ll+lambda).^2);
%     value = 0.5*trace((y-A)'*(y-A)) + 0.5*lambda*sum((m1.^2).*D,'all');
    value = 0.5*trace((y-A)'*(y-A))^2;
    
end