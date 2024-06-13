function d = kernel_cosine(adjmat,dim,mu_v,gamma_v)

% Calculates the link indicator kernel from a graph adjacency by cosine similiarity 
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
		
	n_obj=size(net_y,1);
	W = zeros(n_obj,n_obj);
	for i=1:n_obj
		for j=i:n_obj
				Profile1 = net_y(i,:);
				Profile2 = net_y(j,:);
				sim_v=kernel_to_sim_cos(Profile1,Profile2);
				W(i,j) = sim_v;
				W(j,i) = sim_v;
		end
	end
else
	n_obj=size(net_y,2);
	W = zeros(n_obj,n_obj);
	for i=1:n_obj
		for j=i:n_obj
				Profile1 = net_y(:,i);
				Profile2 = net_y(:,j);
				sim_v=kernel_to_sim_cos(Profile1,Profile2);
				W(i,j) = sim_v;
				W(j,i) = sim_v;
		end
	end
	
end
d=W;
%d = dn(W,'ave');


end



%cosine similiarity 
function sim_cos=kernel_to_sim_cos(v1,v2)
 
	sim_cos = dot(v1,v2)/(norm(v1)*norm(v2));

end