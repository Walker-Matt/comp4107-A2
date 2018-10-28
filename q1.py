import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#target function
def f(X,Y):
    return np.cos((X + 6*(0.35*Y))) + 2*(0.35*X*Y)

trX = np.linspace(-1,1,10)
trY = np.linspace(-1,1,10)
trainX, trainY = np.meshgrid(trX, trY)
trZ = f(trainX,trainY)
trainZ = trZ.ravel()
trainZ = np.vstack(trainZ)

x = trainX.ravel()
y = trainY.ravel()
x = np.vstack(x)
y = np.vstack(y)
trXY = np.concatenate((x,y),axis=1)

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
h_neurons = np.array([2,8,50])
preds = list()
epochs = np.array([], dtype = "int")

for n in h_neurons:
    size_h1 = tf.constant(n, dtype = tf.int32)
    
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])
    
    w_h1 = init_weights([2, size_h1]) # create symbolic variables
    w_o = init_weights([size_h1, 1])
    
    py_x = model(X, w_h1, w_o)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    #cost = tf.norm((py_x-Y), ord=1) ### other possible error function
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) # construct an optimizer
    
    saver = tf.train.Saver()
    
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        print(range(0,len(trXY),batch))
        #for i in range(200):
        error = 10
        i = 0
        while (error > 0.297906):
            for start, end in zip(range(0, len(trXY), batch), range(batch, len(trXY)+1, batch)):
                sess.run(train_op, feed_dict={X: trXY[start:end], Y: trainZ[start:end]})
            predicted = sess.run(py_x, feed_dict={X: teXY})
            error = MSE(testZ,predicted)
            print(i, error)
            trainPred = sess.run(py_x, feed_dict={X: trXY})
            i+=1
        saver.save(sess,"mlp/session.ckpt")
        preds.append(predicted)
        epochs = np.append(epochs,i)

plt.figure()
figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
trContour = plt.contour(testX,testY,teZ,levels = 7, colors = "black")#, label = "target")
two = plt.contour(testX,testY,preds[0].reshape((9,9)),levels = 7, colors = "C0")#, label = "2")
eight = plt.contour(testX,testY,preds[1].reshape((9,9)),levels = 7, colors = "C1")#, label = "8")
fifty = plt.contour(testX,testY,preds[2].reshape((9,9)),levels = 7, colors = "C2")#, label = "50")

print("Number of epochs")
print("2 Hidden Neurons: ", epochs[0])
print("8 Hidden Neurons: ", epochs[1])
print("50 Hidden Neurons: ", epochs[2])
