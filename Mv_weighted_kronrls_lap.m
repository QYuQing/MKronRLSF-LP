function [y_star,weight,weight_lap_1,weight_lap_2 ] = Mv_weighted_kronrls_lap(y_train,K1_list,K2_list,K1_knn_list,K2_knn_list,lambda_list,miu,beta,sita,e,iteration)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

%%init
[~,~,len_k1] = size(K1_list);
[~,~,len_k2] = size(K2_list);
count = 0;
 pre_list = [];value_list = [];y_wknkn_list = [];
for i=1:len_k1
    for j=1:len_k2
        eta_v=0.9;k_nn=100;
        [y_wknkn] = wknkn(y_train,Knormalized(K1_list(:,:,i)),Knormalized(K2_list(:,:,j)),k_nn,eta_v);
        [pre,value] = kronrls(K1_list(:,:,i),K2_list(:,:,j),y_wknkn, lambda_list(i,j));
        value_list = [value_list,value];
        pre = real(pre);
        count = count + 1;
        pre_list(:,:,count) = pre;
        y_wknkn_list(:,:,count) = y_wknkn;
    end
end

Q_1_list = [];C_1_list = [];Q_2_list=[];C_2_list=[];
for i=1:len_k1
    k1 = K1_knn_list(:,:,i);
    d1 = diag(sum(k1,1).^(-0.5));
    w1 = d1*k1*d1;
    [Qa,la] = eig(w1);
    Q_1_list(:,:,i) = Qa;
    C_1_list(:,:,i) = la;
end
for i=1:len_k2
    k2 = K2_knn_list(:,:,i);
    d2 = diag(sum(k2,1).^(-0.5));
    w2 = d2*k2*d2;
    [Qa,la] = eig(w2);
    Q_2_list(:,:,i) = Qa;
    C_2_list(:,:,i) = la;
end
weight_lap_1 = ones(1,len_k1)/len_k1;
weight_lap_2 = ones(1,len_k2)/len_k2;
K1_lap_com = combine_kernels(weight_lap_1,K1_knn_list);
K2_lap_com = combine_kernels(weight_lap_2,K2_knn_list);
weight = ones(1,len_k1*len_k2)/(len_k1*len_k2);
pre_comm = combine_pre(pre_list,weight);
y_star = kronrls_lap(K1_lap_com,K2_lap_com,pre_comm, sita);

%%iter
for iter=1:iteration
    %update weight_lap
    [weight_lap_1] = update_weight_lap_1(Q_1_list,C_1_list,K2_lap_com,y_star,len_k1,e);
    weight_lap_1 = real(weight_lap_1);
    K1_lap_com = combine_kernels(weight_lap_1,K1_knn_list);
    weight_lap_2 = update_weight_lap_2(Q_2_list,C_2_list,K1_lap_com,y_star,len_k2,e);
    weight_lap_2 = real(weight_lap_2);
    K2_lap_com = combine_kernels(weight_lap_2,K2_knn_list);
    %update y_v
    for i=1:len_k1
    for j=1:len_k2
        weight_index = (i-1)*len_k1+j;
        y_other = y_star;
        for k=1:len_k1*len_k2
            if k==weight_index
                continue
            else
                y_other = y_other-weight(k)*pre_list(:,:,k);
            end
        end
        F_new = (1/( 1 + miu * weight(weight_index) )).*y_other + ((miu * weight(weight_index))/( 1 + miu * weight(weight_index) )).*y_wknkn_list(:,:,weight_index);
        F_new = max(F_new,y_train);
        [pre,value] = kronrls(K1_list(:,:,i),K2_list(:,:,j),F_new, lambda_list(i,j));
        value_list(weight_index) = value;
        pre = real(pre);
        pre_list(:,:,weight_index) = pre;
    end
    end

    %update weight
    y_star = real(y_star);
    [weight] = compute_comm_weight(y_star,pre_list,value_list,beta,miu);
    pre_comm = combine_pre(pre_list,weight);
    pre_comm = max(pre_comm,y_train);
    %update y*
    y_star = kronrls_lap(K1_lap_com,K2_lap_com,pre_comm, sita);
    y_star = real(y_star);
%     [~,~,~,aupr_kronrls] = perfcurve(y(test_idx),y_star(test_idx),1, 'xCrit', 'reca', 'yCrit', 'prec');
%     fprintf('- iter %d - AUPR: %f \n', iter, aupr_kronrls)
end


end


function [weight] = update_weight_lap_1(Q_1_list,C_1_list,K2_com,y,len_k1,e)
d_com = diag(sum(K2_com,1).^(-0.5));
w_com = d_com*K2_com*d_com;
[Q_com,C_com] = eig(w_com);

fenzi_value_list = [];
for i=1:len_k1
    Q_1 = Q_1_list(:,:,i);
    C_1 = C_1_list(:,:,i);
    fenzi_value = (Q_1'*y*Q_com).*(kron(diag(C_1),diag(C_com)')).*(Q_1'*y*Q_com);
    fenzi_value = sum(fenzi_value,'all');
    fenzi_value_list = [fenzi_value_list,fenzi_value];
end
weight = (1./fenzi_value_list).^(1/(e-1));
weight = weight./sum(weight,'all');
weight = weight.^e;
end

function [weight] = update_weight_lap_2(Q_2_list,C_2_list,K1_com,y,len_k2,e)
d_com = diag(sum(K1_com,1).^(-0.5));
w_com = d_com*K1_com*d_com;
[Q_com,C_com] = eig(w_com);

fenzi_value_list = [];
for i=1:len_k2
    Q_2 = Q_2_list(:,:,i);
    C_2 = C_2_list(:,:,i);
    fenzi_value = (Q_com'*y*Q_2).*(kron(diag(C_com),diag(C_2)')).*(Q_com'*y*Q_2);
    fenzi_value = sum(fenzi_value,'all');
    fenzi_value_list = [fenzi_value_list,fenzi_value];
end
weight = (1./fenzi_value_list).^(1/(e-1));
weight = weight./sum(weight,'all');
weight = weight.^e;
end

function [weight] = compute_comm_weight(y_star,pre_list,value_list,beta,miu)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
[~,~,v] = size(pre_list);
Q = zeros(v,v);
for i=1:v
    for j=i:v
        Q(i,j) = 0.5*trace(pre_list(:,:,i)'*pre_list(:,:,j));
        Q(j,i) = Q(i,j);
    end
end
Q = Q./(sum(Q.^2,1).^0.5);
Q = (Q+Q')./2;

Q = Q + beta*eye(v);
            
y_star_list = [];
for k=1:v
    y_star_list = [y_star_list , trace(y_star'*pre_list(:,:,k))];
end
q = miu*value_list-y_star_list;
q = q./(sum(q.^2,'all')^0.5);
q = double(q);

%x = quadprog(H,f,A,b,Aeq,beq,lb,ub)
%min x 0.5*x'H*x+f'*x  A*x<=b; Aeq*x=beq; lb<=x<=ub;
weight = quadprog(Q,q,[],[],ones(1,v),1,zeros(1,v),ones(1,v));
end

