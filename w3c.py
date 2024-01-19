import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(100):
    eq7 = sigmoid(i)*(1-sigmoid(i))
    fin_d = (sigmoid(i + (10 ** -5)/2)-sigmoid(i - (10 ** -5)/2)) / 10 ** -5
    zeroCheck = eq7 - fin_d
    print(zeroCheck)