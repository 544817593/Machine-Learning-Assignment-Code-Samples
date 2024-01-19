import numpy as np
import matplotlib.pyplot as plt

N = int(1e4); D = 2
a = 1
A = [[2,0],[0,5]]
X = A @ np.random.randn(D,N)
plt.plot(X[0,:], X[1,:], '.')

plt.axis('square')
plt.show()

# D = 3; Sigma = np.cov(np.random.randn(D, 3*D))
# print(Sigma)
# A = np.linalg.cholesky(Sigma)
# Sigma_from_A = A @ A.T  # up to round-off error, matches Sigma

# print(Sigma_from_A)


# N = 1000; D = 2
# X = np.random.randn(N, D)
# def plot_points(aa):
#     A = np.array([[1, 0], [aa, 1-aa]])
#     Z = np.dot(X, A.T) 
#     plt.clf()
#     plt.plot(Z[:,0], Z[:,1], '.')
#     plt.axis('square')
#     plt.show()
# plot_points(0.1)