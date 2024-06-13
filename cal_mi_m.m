function [mi_kernel] = cal_mi_m(adjmat,dim)

net_y = adjmat;
if dim == 1
		
	n_obj=size(net_y,1);
	W = zeros(n_obj,n_obj);
	for i=1:n_obj
		for j=i:n_obj
				Profile1 = net_y(i,:);
				Profile2 = net_y(j,:);
				sim_v = mutualinfo(Profile1,Profile2);
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
				sim_v = mutualinfo(Profile1,Profile2);
				W(i,j) = sim_v;
				W(j,i) = sim_v;
		end
	end
	
end
mi_kernel=W;


