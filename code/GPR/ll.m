function [J grad] = ll(theta, X,y,n)
    Ky = exp_cov(X,X,theta)+(theta(3)^2)*eye(n);
     Kyi = inv(Ky);
    J = (1/2)*y'*Kyi*y+(1/2)*log(det(Ky))+(n/2)*log(2*pi);
    
    grad = zeros(3,1);
    alpha = Ky\y;
    al = alpha*alpha';
    A = al-Kyi;
    grad(1) =(1/2)*trace(A*(2/theta(1))*Ky);
    d = pdist2(X,X);
    d2 = d.^2;
    d2l = d2/(theta(2)^3);
    grad(2) =(1/2)*trace(A*Ky*d2l);
    grad(3) =(1/2)*trace(A*(2/theta(3))*eye(n));
end