# Softmax activation function


import numpy as np

L = np.array([1,2,3,4,5,6])

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

softmax(L)
