import numpy as np
import matplotlib.pyplot as plt

########### WEEK 2 ##########

amp_data = np.load('amp_data.npz')['amp_data']

# 2.1.a
# plt.figure(figsize=(15,10))

# plt.subplot(2,1,1)
# plt.plot(amp_data)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title("Lineplot for amp_data")

# plt.subplot(2,1,2)
# plt.hist(amp_data,6000, range =  [-0.4,0.4])
# plt.xlabel('Amplitude')
# plt.ylabel('Frequency')
# plt.title("Barplot for amp_data")

# plt.show()

# 2.1.b
amp_data_matrix = np.reshape(amp_data[0:33713274],[1605394,21])
np.random.seed(7)
shuffled_amp_data_matrix = np.random.permutation(amp_data_matrix)

X_shuf_train = shuffled_amp_data_matrix[0:1123776][:,:20]
y_shuf_train = shuffled_amp_data_matrix[0:1123776][:,-1:]
X_shuf_val = shuffled_amp_data_matrix[1123776:1364586][:,:20]
y_shuf_val = shuffled_amp_data_matrix[1123776:1364586][:,-1:]
# X_shuf_test = shuffled_amp_data_matrix[1364586:1605394][:20,:]
X_shuf_test = shuffled_amp_data_matrix[1364586:1605394][:,:20]
y_shuf_test = shuffled_amp_data_matrix[1364586:1605394][:,-1:]

# 2.2.a

def linear(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def phi_quadratic(X):
    return np.hstack([np.ones((X.shape[0],1)), X, X**2, X**3, X**4])

# Training data points
X = np.arange(0,1, step = 0.05).reshape(20,1)
yy = X_shuf_train[1,:]

w = np.linalg.lstsq(linear(X), yy,rcond=None)[0]
quad_w = np.linalg.lstsq(phi_quadratic(X),yy,rcond=None)[0]

plt.figure(figsize=(15,7))
plt.scatter(X,yy,marker='x',label="Training points")
plt.scatter(1,y_shuf_train[1,:],marker='x',color='r',label="Correct prediction point")

Prediction_input = np.arange(0,1.05,step=0.05).reshape(21,1)
plt.plot(Prediction_input, linear(Prediction_input) @ w,color='b',label='Linear')
plt.plot(Prediction_input, phi_quadratic(Prediction_input) @ quad_w,color='g',label='Quadratic')

plt.xticks(np.arange(0,1.05,step=0.05))
plt.xlabel('Time')
plt.ylabel('Amplitude')
#plt.legend()
plt.title('Amplitude prediction for two models')
#plt.show()







########### WEEK 3 ##########

def Phi(C,K):
    matrix = np.empty([C,K],dtype = float)
    for i in range(C):
        t = 1 - 0.05*(i+1)
        eachRow = np.zeros(K)
        for j in range(K):
            eachRow[j] = t ** j
        matrix[C-1-i] = eachRow
    return matrix

def make_vv(C,K):
    phiTphi = np.dot(Phi(C,K).T,Phi(C,K))
    phiTphiInv = np.linalg.inv(phiTphi)
    v = np.dot(Phi(C,K),phiTphiInv)
    return np.dot(v,np.ones((K,1)))

linear_v = make_vv(20,2)
linear_pred = np.dot(linear_v.T,yy)

quartic_v = make_vv(20,5)
quartic_pred = np.dot(quartic_v.T,yy)

plt.scatter(1,linear_pred,marker='o',label='vTx Linear') 
plt.scatter(1,quartic_pred,marker='v',label='vTx Quartic')
plt.legend()
# plt.show()


def FindBestCK():
    min_sse_c_k = np.array([500000,0,0])
    for c in range(1,21):
        for k in range(1,10):
            sse = 0    
            temp_v = make_vv(c,k)
            for x in range(len(X_shuf_train)):
                temp_pred = np.dot(temp_v.T,X_shuf_train[x,-c:])
                temp_sq_error = (y_shuf_train[x,:] - temp_pred) ** 2
                sse += temp_sq_error
            if sse < min_sse_c_k[0]:
                min_sse_c_k = [sse,c,k]
            print("c = " + str(c) + "    k = " + str(k) + "   sse = " + str(sse))    
    return min_sse_c_k[1],min_sse_c_k[2]

# m = FindBestCK()
# print(m)


def EvaluateSystem(c,k):
    mse_train = 0
    mse_val = 0
    mse_test = 0
    v = make_vv(c,k)

    for x in range(len(X_shuf_train)):
        temp_pred = np.dot(v.T,X_shuf_train[x,-c:])
        temp_sq_error = (y_shuf_train[x,:] - temp_pred) ** 2
        mse_train += temp_sq_error
    mse_train /= len(X_shuf_train)

    for x in range(len(X_shuf_val)):
        temp_pred = np.dot(v.T,X_shuf_val[x,-c:])
        temp_sq_error = (y_shuf_val[x,:] - temp_pred) ** 2
        mse_val += temp_sq_error
    mse_val /= len(X_shuf_val)

    for x in range(len(X_shuf_test)):
        temp_pred = np.dot(v.T,X_shuf_test[x,-c:])
        temp_sq_error = (y_shuf_test[x,:] - temp_pred) ** 2
        mse_test += temp_sq_error
    mse_test /= len(X_shuf_test)

    return mse_train, mse_val, mse_test

# m = EvaluateSystem(1,1)
# print(m)
    

def Q2a():
    for c in range(1,21):
        mse_train = 0
        v = np.linalg.lstsq(X_shuf_train[:,-c:], y_shuf_train,rcond=None)[0]
        for x in range(len(X_shuf_train)):
            temp_pred = np.dot(v.T,X_shuf_train[x,-c:])
            temp_sq_error = (y_shuf_train[x,:] - temp_pred) ** 2
            mse_train += temp_sq_error
        mse_train /= len(X_shuf_train)

        mse_val = 0
        for x in range(len(X_shuf_val)):
            temp_pred = np.dot(v.T,X_shuf_val[x,-c:])
            temp_sq_error = (y_shuf_val[x,:] - temp_pred) ** 2
            mse_val += temp_sq_error
        mse_val /= len(X_shuf_val)        

        print("Training MSE with c = " + str(c) + " : " + str(mse_train))
        print("Validation MSE with c = " + str(c) + " : " + str(mse_val))

# Q2a()

def Q2b():
    v_poly = make_vv(2,2)
    v_q2 = np.linalg.lstsq(X_shuf_train[:,-19:], y_shuf_train,rcond=None)[0]
    mse_poly = 0
    mse_q2 = 0
    for x in range(len(X_shuf_test)):
        poly_temp_pred = np.dot(v_poly.T,X_shuf_test[x,-2:])
        q2_temp_pred = np.dot(v_q2.T,X_shuf_test[x,-19:])

        poly_temp_sq_error = (y_shuf_test[x,:] - poly_temp_pred) ** 2
        q2_temp_sq_error = (y_shuf_test[x,:] - q2_temp_pred) ** 2

        mse_poly += poly_temp_sq_error
        mse_q2 += q2_temp_sq_error
    mse_poly /= len(X_shuf_test)
    mse_q2 /= len(X_shuf_test)
    print("Q1 predictor MSE: " + str(mse_poly))
    print("Q2 predictor MSE: " + str(mse_q2))

# Q2b()


def Q3():
    splitted_data = np.vsplit(shuffled_amp_data_matrix[0:1364586],6)
    best_cs = np.zeros(6)

    y_val = splitted_data[0][:,-1:]
    x_val = splitted_data[0][:,:20]
    train = np.concatenate(splitted_data[1:6])
    x_train = train[:,:20]
    y_train = train[:,-1:]
    best_cs[0] = FindBestC(x_train, y_train,x_val,y_val)

    y_val = splitted_data[1][:,-1:]
    x_val = splitted_data[1][:,:20]
    train = np.concatenate(splitted_data[0] + splitted_data[2:6])
    x_train = train[:,:20]
    y_train = train[:,-1:]
    best_cs[1] = FindBestC(x_train, y_train,x_val,y_val)

    y_val = splitted_data[2][:,-1:]
    x_val = splitted_data[2][:,:20]
    train = np.concatenate(splitted_data[0:2] + splitted_data[3:6])
    x_train = train[:,:20]
    y_train = train[:,-1:]
    best_cs[2] = FindBestC(x_train, y_train,x_val,y_val)

    y_val = splitted_data[3][:,-1:]
    x_val = splitted_data[3][:,:20]
    train = np.concatenate(splitted_data[0:3] + splitted_data[4:6])
    x_train = train[:,:20]
    y_train = train[:,-1:]
    best_cs[3] = FindBestC(x_train, y_train,x_val,y_val)

    y_val = splitted_data[4][:,-1:]
    x_val = splitted_data[4][:,:20]
    train = np.concatenate(splitted_data[0:4] + splitted_data[5:6])
    x_train = train[:,:20]
    y_train = train[:,-1:]
    best_cs[4] = FindBestC(x_train, y_train,x_val,y_val)

    y_val = splitted_data[5][:,-1:]
    x_val = splitted_data[5][:,:20]
    train = np.concatenate(splitted_data[0:5])
    x_train = train[:,:20]
    y_train = train[:,-1:]
    best_cs[5] = FindBestC(x_train, y_train,x_val,y_val)

    optimal_c_val = int(round(np.mean(best_cs)))
    print (optimal_c_val)


    v = np.linalg.lstsq(X_shuf_train[:,-optimal_c_val:], y_shuf_train,rcond=None)[0]

    mse = 0
    for x in range(len(X_shuf_test)):
        temp_pred = np.dot(v.T,X_shuf_test[x,-optimal_c_val:])
        temp_sq_error = (y_shuf_test[x,:] - temp_pred) ** 2
        mse += temp_sq_error
    mse /= len(X_shuf_test)
    print(mse)



def FindBestC(x_train, y_train, x_val, y_val):
    best_c = -1
    min_mse_val = 99999
    for c in range(12,21):
        mse_train = 0
        v = np.linalg.lstsq(x_train[:,-c:], y_train,rcond=None)[0]
        for x in range(len(x_train)):
            temp_pred = np.dot(v.T,x_train[x,-c:])
            temp_sq_error = (y_train[x,:] - temp_pred) ** 2
            mse_train += temp_sq_error
        mse_train /= len(x_train)

        mse_val = 0
        for x in range(len(x_val)):
            temp_pred = np.dot(v.T,x_val[x,-c:])
            temp_sq_error = (y_val[x,:] - temp_pred) ** 2
            mse_val += temp_sq_error
        mse_val /= len(x_val)      

        if (mse_val < min_mse_val):
            min_mse_val = mse_val
            best_c = c
        print(c)
    return best_c

Q3()

