import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import timeit

#target function
def f(X,Y):
    return np.cos((X + 6*(0.35*Y))) + 2*(0.35*X*Y)

#10 values from -1 to 1
trX = np.linspace(-1,1,10)
trY = np.linspace(-1,1,10)

trainX, trainY = np.meshgrid(trX, trY)  #10 x 10 matrix for contour plot
trZ = f(trainX,trainY)  #z values for contour plot

trainZ = trZ.ravel()
trainZ = np.vstack(trainZ)  #100 z values in one column

#produces 100 rows of x,y pairs (2 columns)
x = trainX.ravel()
y = trainY.ravel()
x = np.vstack(x)
y = np.vstack(y)
trXY = np.concatenate((x,y),axis=1)

#same process as above, but for test values
teX = np.linspace(-1,1,9)
teY = np.linspace(-1,1,9)
testX, testY = np.meshgrid(teX, teY)
teZ = f(testX,testY)
testZ = teZ.ravel()
testZ = np.vstack(testZ)
x = testX.ravel()
y = testY.ravel()
x = np.vstack(x)
y = np.vstack(y)
teXY = np.concatenate((x,y),axis=1)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_o):
    #h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    h1 = tf.nn.tanh(tf.matmul(X, w_h1))
    return tf.matmul(h1, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def MSE(A,B):
    return np.square(A-B).mean(axis=0)

batch = 1
preds = list()
numEpochs = np.array([], dtype = "int")
epochs = np.linspace(1,100,100)
allMSE = list()
CPU = list()

size_h1 = tf.constant(8, dtype = tf.int32)

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

w_h1 = init_weights([2, size_h1]) # create symbolic variables
w_o = init_weights([size_h1, 1])

py_x = model(X, w_h1, w_o)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
#cost = tf.reduce_mean(tf.norm((py_x-Y), ord=1)) ### other possible error function
cost = tf.nn.tanh(tf.losses.mean_squared_error(labels = Y, predictions=py_x))
train_op1 = tf.train.GradientDescentOptimizer(0.02).minimize(cost) # construct an optimizer
train_op2 = tf.train.MomentumOptimizer(0.02,0.02).minimize(cost) # construct an optimizer
train_op3 = tf.train.RMSPropOptimizer(0.02).minimize(cost) # construct an optimizer

saver = tf.train.Saver()

mse = np.array([])
time = np.array([])
for e in epochs:
    start = timeit.timeit()
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        #print(range(0,len(trXY),batch))
        #for i in range(200):
        error = 100
        i = 0
        while (error > 0.02 and i<e):
            for start, end in zip(range(0, len(trXY), batch), range(batch, len(trXY)+1, batch)):
                sess.run(train_op1, feed_dict={X: trXY[start:end], Y: trainZ[start:end]})
            predicted = sess.run(py_x, feed_dict={X: teXY})
            error = MSE(testZ,predicted)
            print(i, error)
            trainPred = sess.run(py_x, feed_dict={X: trXY})
            i+=1
        end = timeit.timeit()
        diff = end - start
        saver.save(sess,"mlp/session.ckpt")
        preds.append(predicted)
        epochs = np.append(epochs,i)
        mse = np.append(mse,error)
        sess.close()
allMSE.append(mse)
CPU.append(time)

mse = np.array([])
time = np.array([])
for e in epochs:
    start = timeit.timeit()
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        #print(range(0,len(trXY),batch))
        #for i in range(200):
        error = 100
        i = 0
        while (error > 0.02 and i <= e):
            for start, end in zip(range(0, len(trXY), batch), range(batch, len(trXY)+1, batch)):
                sess.run(train_op2, feed_dict={X: trXY[start:end], Y: trainZ[start:end]})
            predicted = sess.run(py_x, feed_dict={X: teXY})
            error = MSE(testZ,predicted)
            print(i, error)
            trainPred = sess.run(py_x, feed_dict={X: trXY})
            i+=1
        end = timeit.timeit()
        diff = end - start
        saver.save(sess,"mlp/session.ckpt")
        preds.append(predicted)
        epochs = np.append(epochs,i)
        mse = np.append(mse,error)
        sess.close()
allMSE.append(mse)
CPU.append(time)

mse = np.array([])
time = np.array([])
for e in epochs:
    start = timeit.timeit()
# Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        #print(range(0,len(trXY),batch))
        #for i in range(200):
        error = 100
        i = 0
        while (error > 0.02 and i<1000):
            for start, end in zip(range(0, len(trXY), batch), range(batch, len(trXY)+1, batch)):
                sess.run(train_op3, feed_dict={X: trXY[start:end], Y: trainZ[start:end]})
            predicted = sess.run(py_x, feed_dict={X: teXY})
            error = MSE(testZ,predicted)
            print(i, error)
            trainPred = sess.run(py_x, feed_dict={X: trXY})
            i+=1
        end = timeit.timeit()
        diff = end - start
        saver.save(sess,"mlp/session.ckpt")
        preds.append(predicted)
        epochs = np.append(epochs,i)
        mse = np.append(mse,error)
        sess.close()
allMSE.append(mse)
CPU.append(time)