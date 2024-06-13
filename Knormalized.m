function S=Knormalized(K)
%kernel normilization
K = abs(K);
kk = K(:);
kk(find(kk==0)) = [];
min_v = min(kk);
K(find(K==0))=min_v;

D=diag(K);
D=sqrt(D);
S=K./(D*D');

end