import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

ker = [4.0*kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),  1.0*kernels.RBF(length_scale=0.2, length_scale_bounds=(1e-1, 10.0)), 0.25*kernels.RBF(length_scale=4.0, length_scale_bounds=(1e-1, 10.0))]

data = np.random.rand(4,1)*5
y = 2*np.exp(data)
for k in ker:
	plt.figure()
	gp = GaussianProcessRegressor(kernel=k)

	gp.fit(data,y)

	plt.scatter(data,y)

	X = np.linspace(0,5,100)
	y_mean, y_std = gp.predict(X[:,np.newaxis], return_std=True)
	y_mean = y_mean[:,0]
	plt.plot(X,y_mean)
	plt.fill_between(X,y_mean-y_std,y_mean+y_std,alpha=0.2)

	plt.title("Kernel : %s" % (gp.kernel))

plt.show()