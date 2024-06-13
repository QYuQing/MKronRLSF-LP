function d = kernel_corr(adjmat,dim,mu_v,gamma_v)

% Calculates the link indicator kernel from a graph adjacency by pairwise linear correlation coefficient
% 
%INPUT: 
% adjmat : binary adjacency matrix
% dim    : dimension (1 - rows, 2 - cols)
%OUTPUT:
% d : kernel matrix for adjmat over dimension 'dim'

 net_y = adjmat;
% Graph based kernel

%Gaussian random noise matrix
R = normrnd(mu_v,gamma_v,size(net_y));
%Add noise
net_y = net_y + R;
if dim == 1
	
	net_y = net_y';
	W = corr(net_y);
	
else
	W = corr(net_y);
	
end
d=W;
%d = dn(W,'ave');


end

