import tensorflow as tf
import numpy as np

#target function
def f(X, Y):
    return np.cos((X + 6*(0.35*Y))) + 2*(0.35*X*Y)

train = np.zeros((10,10,2), dtype = 'float') #Training set
test = np.zeros((9,9,2), dtype = 'float') #Testing set

#Populating the training set (10x10)
#Ranges from -1 to 0.8
Xval = -1
for i in range(10):
    Yval = -1
    for j in range(10):
        pair = [Xval, Yval]
        train[i][j] = pair
        Yval = Yval + 0.2
    Xval = Xval + 0.2
    
#Populating the testing set (9x9)
#Ranges from -0.8 to 0.8
Xval = -0.8
for i in range(9):
    Yval = -0.8
    for j in range(9):
        pair = [Xval, Yval]
        test[i][j] = pair
        Yval = Yval + 0.2
    Xval = Xval + 0.2
    
