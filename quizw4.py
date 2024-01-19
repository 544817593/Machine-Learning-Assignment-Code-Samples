import numpy as np

X = np.random.randn(3,1)

np.random.seed(777)
Z = np.random.randn(4,3)
A = np.cov(Z,rowvar=False) #3,3
C = np.random.randn(3,1)

mu = (-0.5*np.linalg.inv(A) @ C) #3,1
cov = 0.5*np.linalg.inv(A) #3,3

gaussianPx = np.exp(-0.5 * (X-mu).T @ np.linalg.inv(cov) @ (X-mu))

xtax = X.T @ A @ X
xtc = X.T @ C
poportionalTo = np.exp(-xtax-xtc)

print(gaussianPx/poportionalTo)



