function M = exp_cov(A,B,theta)
    d = pdist2(A,B);
    d2 = d.^2;
    d2l = d2/(2*theta(2));
    d2e = vpa(exp(-d2l));
    M = (theta(1)^2)*d2e;
end
% for both 1d and Nd regression