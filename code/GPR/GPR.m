theta_0 = [1;10;1];
n = size(X,1);
[theta fx c] = minimize(theta_0, 'll', 25, X, y, n); 
testdata = linspace(0,100)';
K = exp_cov(X,X,theta);
Ks = exp_cov(testdata,X,theta);
Kss = exp_cov(testdata,testdata,theta);

mu = Ks*((K+theta(3)*eye(n))\y);
cov = Kss - Ks*((K+theta(3)*eye(n))\Ks');
sig = diag(cov);
% plot points and function with error bars
% for 1d regression
plot(X,y,'Marker',"*");
hold on;
errorbar(testdata, mu, sig);
hold off;
