function [y_pre] = kronrls_weighted_Lap(K1_list,K2_list,y,sita,e,iteration)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
[~,~,len_k1] = size(K1_list);
[~,~,len_k2] = size(K2_list);

Q_1_list = [];C_1_list = [];Q_2_list=[];C_2_list=[];
for i=1:len_k1
    k1 = K1_list(:,:,i);
    d1 = diag(sum(k1,1).^(-0.5));
    w1 = d1*k1*d1;
    [Qa,la] = eig(w1);
    Q_1_list(:,:,i) = Qa;
    C_1_list(:,:,i) = la;
end

for i=1:len_k2
    k2 = K2_list(:,:,i);
    d2 = diag(sum(k2,1).^(-0.5));
    w2 = d2*k2*d2;
    [Qa,la] = eig(w2);
    Q_2_list(:,:,i) = Qa;
    C_2_list(:,:,i) = la;
end
%init
weight_1 = ones(1,len_k1)/len_k1;
weight_2 = ones(1,len_k2)/len_k2;
K1_com = combine_kernels(weight_1,K1_list);
K2_com = combine_kernels(weight_2,K2_list);

for iter=1:iteration
    
    [y_pre] = kronrls_lap(K1_com,K2_com,y, sita);
    [weight_1] = update_weight_1(Q_1_list,C_1_list,K2_com,y_pre,len_k1,e);
    K1_com = combine_kernels(weight_1,K1_list);
    [weight_2] = update_weight_2(Q_2_list,C_2_list,K1_com,y_pre,len_k2,e);
    K2_com = combine_kernels(weight_2,K2_list);
end

end

function [weight] = update_weight_1(Q_1_list,C_1_list,K2_com,y,len_k1,e)
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

function [weight] = update_weight_2(Q_2_list,C_2_list,K1_com,y,len_k2,e)
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

