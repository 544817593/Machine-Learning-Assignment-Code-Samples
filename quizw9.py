from w9_support import *
import numpy as np
from scipy.optimize import minimize

# Q2a
# ct_data = np.load('ct_data.npz')
# X_train = ct_data['X_train']
# X_val = ct_data['X_val']
# X_test = ct_data['X_test']
# y_train = ct_data['y_train']
# y_val = ct_data['y_val']
# y_test = ct_data['y_test']

# np.random.seed(77)

# def fit_nn(X, yy, alpha):
#     K = 10
#     D = X.shape[1]
#     args = (X, yy, alpha)
#     init = (0.1*np.random.randn(K,)/np.sqrt(K), 0.1*np.random.randn()/np.sqrt(K), 0.1*np.random.randn(K, D)/np.sqrt(K), 0.1 * np.random.randn(K,)/np.sqrt(K))
#     ww, bb, V, bk = minimize_list(nn_cost, init, args)
#     return (ww, bb, V, bk)

# params = fit_nn(X_train, y_train, 10)
# E, (ww_bar, bb_bar, V_bar, bk_bar) = nn_cost(params, X_train, y_train, alpha=0)
# rmse = (E / X_train.shape[0]) ** 0.5
# print(rmse)

# Q2b
import numpy as np
from w9_support import *

data = np.load('ct_data.npz')
y_train = data['y_train']
X_train = data['X_train']
D = X_train.shape[1]

def rmse(ff, yy):
    return np.sqrt(np.mean((ff - yy)**2))

def sigmoid(aa):
    return 1. / (1. + np.exp(-aa))

def transform(X, W1, bb1):
    return sigmoid((X @ W1) + bb1)

def fit_logreg_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

K = 10
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
ww1 = np.zeros((D, K))
bb1 = np.zeros(K)
for kk in range(K):
    labels = y_train > thresholds[kk]
    ww1[:,kk], bb1[kk] = fit_logreg_gradopt(X_train, labels, alpha=10)

Z = transform(X_train, ww1, bb1)
ww2, bb2 = fit_linreg_gradopt(Z, y_train, alpha=10)

def fit_nn(X, yy, alpha):
    K = 10
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (ww2, bb2, ww1.T, bb1)
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    return (ww, bb, V, bk)

params = fit_nn(X_train, y_train, 10)
E, (ww_bar, bb_bar, V_bar, bk_bar) = nn_cost(params, X_train, y_train, alpha=0)
rmse = (E / X_train.shape[0]) ** 0.5

# E, (ww_bar, bb_bar, V_bar, bk_bar) = nn_cost(params, X_val, y_val, alpha=0)
# rmse = (E / X_val.shape[0]) ** 0.5

print(rmse)