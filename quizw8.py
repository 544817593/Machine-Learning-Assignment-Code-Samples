from w8_support import *
import numpy as np
from scipy.optimize import minimize

# Q1a
ct_data = np.load('ct_data.npz')
X_train = ct_data['X_train']
X_val = ct_data['X_val']
X_test = ct_data['X_test']
y_train = ct_data['y_train']
y_val = ct_data['y_val']
y_test = ct_data['y_test']

(ww,bb) = fit_linreg_gradopt(X_train, y_train, 10)

train_sse = 0
for i in range(len(X_train)):
    prediction = ww.T @ X_train[i] + bb
    sq_error = (y_train[i] - prediction) ** 2
    train_sse += sq_error
train_rmse = (train_sse / len(X_train)) ** 0.5

val_sse = 0
for i in range(len(X_val)):
    prediction = ww.T @ X_val[i] + bb
    sq_error = (y_val[i] - prediction) ** 2
    val_sse += sq_error
val_rmse = (val_sse / len(X_val)) ** 0.5

print("Training RMSE: " + str(train_rmse) + "\n" + "Validation RMSE: " + str(val_rmse))

# Q1b
def fit_logreg_gradopt(X, yy, alpha):
    D = X.shape[1]
    args = (X, yy, alpha)
    init = (np.zeros(D), np.array(0))
    ww, bb = minimize_list(logreg_cost, init, args)
    return ww, bb

transformed_X_train = np.zeros((len(X_train), 10))
transformed_X_val = np.zeros((len(X_val), 10))

K = 10 # number of thresholded classification problems to fit
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
for kk in range(K):
    labels = y_train > thresholds[kk]
    # ... fit logistic regression to these labels
    (ww, bb) = fit_logreg_gradopt(X_train, labels, 10)

    wtx_b_train = ww.T @ X_train.T + bb
    predictions_train = 1/(1+np.exp(wtx_b_train))

    wtx_b_val = ww.T @ X_val.T + bb
    predictions_val = 1/(1+np.exp(wtx_b_val))

    transformed_X_train[:,kk] = predictions_train
    transformed_X_val[:,kk] = predictions_val

(ww,bb) = fit_linreg_gradopt(transformed_X_train, y_train, 10)

train_sse = 0
for i in range(len(transformed_X_train)):
    prediction = ww.T @ transformed_X_train[i] + bb
    sq_error = (y_train[i] - prediction) ** 2
    train_sse += sq_error
train_rmse = (train_sse / len(transformed_X_train)) ** 0.5

val_sse = 0
for i in range(len(transformed_X_val)):
    prediction = ww.T @ transformed_X_val[i] + bb
    sq_error = (y_val[i] - prediction) ** 2
    val_sse += sq_error
val_rmse = (val_sse / len(transformed_X_val)) ** 0.5

print("Training RMSE: " + str(train_rmse) + "\n" + "Validation RMSE: " + str(val_rmse))


    
