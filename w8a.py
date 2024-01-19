import numpy as np
from matplotlib import pyplot as plt

# def neural_net(X):
#     h1 = np.zeros(X.shape)
#     for i in range(100):
#         ww = np.random.normal(0, 1,(1,len(X))) # 1,N
#         h1[i] = 1 / (1 + np.exp(-(ww @ X)))

#     h2 = np.zeros((50,1))
#     for i in range(50):
#         ww = np.random.normal(0, 1,(1,len(h1))) # 1,100
#         h2[i] = 1 / (1 + np.exp(-(ww @ X + i)))

#     ww = np.random.normal(0, 1,(1,len(h2))) # 1,50
#     ff = 1 / (1 + np.exp(-(ww @ h2)))
#     return ff

def sigmoid(a): return 1. / (1. + np.exp(-a))
def relu(x): return np.maximum(x, 0)
def linear(a): return a

def neural_net(X, layer_sizes=(100,50,1), gg=sigmoid, sigma_w=1):
    for out_size in layer_sizes:
        Wt = sigma_w * np.random.randn(X.shape[1], out_size)
        X = gg(X @ Wt)
        print(X[1])
    return X

N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1
plt.clf()
for i in range(12):
    ff = neural_net(X)
    plt.plot(X, ff)

plt.show()