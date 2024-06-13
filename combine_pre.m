function [pre_comm] = combine_pre(pre_list,weight)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
[n,m,l] = size(pre_list);
pre_comm = zeros(n,m);
for i=1:l
    pre_comm = pre_comm + weight(i)*pre_list(:,:,i);
end
end

