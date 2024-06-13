function [F_new] = wknkn(Y,W1,W2,k_nn,eta_v)
%tju cs, bioinformatics. This program is recoded by reference follow:
%Weighted K nearest known neighbors (WKNKN)
%ref:
%      Ezzat A, Zhao P, Wu M, et al. 
%      Drug-Target Interaction Prediction with Graph Regularized Matrix Factorization[J]. 
%           IEEE/ACM Transactions on Computational Biology & Bioinformatics, 2016, PP(99):1-1.
% W1 : the kernel of object 1, (m-by-m)
% W2 : the kernel of object 2, (n-by-n)
% Y  : binary adjacency matrix, (m-by-n)
%eta_v: decay term (<1)
%k_nn: the k nearest neighbor samples (30)

%initialize two matrices
[row_s col_s] = size(Y);
Y_d = zeros(row_s, col_s);
Y_t=Y_d;

for d=1:row_s
	dnn = KNearestKnownNeighbors(d,W1,k_nn);
	w_i = zeros(1,k_nn);
	for ii=1:k_nn
		w_i(ii) = (eta_v^(ii-1))*W1(d,dnn(ii));
	end
	%normalization term
	Z_d = [];
	Z_d = W1(d,dnn);
	Z_d = 1/(sum(Z_d));
	
	Y_d(d,:) = Z_d*(w_i*Y(dnn,:));

end


for t = 1:col_s
	tnn =  KNearestKnownNeighbors(t,W2,k_nn);
	w_j = zeros(1,k_nn);
	
	for jj=1:k_nn
		w_j(jj) = (eta_v^(jj-1))*W2(t,tnn(jj));
	end
	%normalization term
	Z_t = [];
	Z_t = W2(t,tnn);
	Z_t = 1/(sum(Z_t));
	
	Y_t(:,t) = Z_t*(Y(:,tnn)*w_j');
	
end

Y_dt = (Y_d+Y_t)/2;

F_new = [];
F_new = max(Y,Y_dt);

end



function similarities_N = KNearestKnownNeighbors(index_i,similar_m,kk)


		iu = similar_m(index_i,:);
		[B, iu_list] = sort(iu,'descend');
		similarities_N = iu_list(1:kk);

end