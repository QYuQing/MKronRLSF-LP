function y2=predict_kernel(y,ka,lambda)

	
	%ka2 = ka - diag(diag(ka));
	%d = diag(sum(ka2,2));
	%y2 = inv(d)*ka2*y;
	k = ka;
	n = size(ka,1);

	
	%t2 = (k ^ 2   + 1 * eye(n)) \ (k * y);
	t2 = (k    + lambda * eye(n)) \ (y);
    y2 = k * t2;
	

end