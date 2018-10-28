import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#target function
def f(X,Y):
    #X = p[0]
    #Y = p[1]
    return np.cos((X + 6*(0.35*Y))) + 2*(0.35*X*Y)

trX = np.linspace(-1,1,10)
trY = np.linspace(-1,1,10)
trainX, trainY = np.meshgrid(trX, trY)
trZ = f(trainX,trainY)
trainZ = trZ.ravel()
trContour = plt.contour(trainX,trainY,trZ,levels = 8)

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
teContour = plt.contour(testX,testY,teZ,levels = 8)

x = testX.ravel()
y = testY.ravel()
x = np.vstack(x)
y = np.vstack(y)
teXY = np.concatenate((x,y),axis=1)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h1, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

size_h1 = tf.constant(8, dtype=tf.int32)

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

w_h1 = init_weights([2, size_h1]) # create symbolic variables
w_o = init_weights([size_h1, 1])

py_x = model(X, w_h1, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.000001).minimize(cost) # construct an optimizer
#predict_op = tf.argmax(py_x, 0)

saver = tf.train.Saver()

def MSE(A,B):
    return ((A - B) ** 2).mean(axis=0)

bach = 10
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    print(range(0,len(trXY),5))
    for i in range(20):
    #error = 10
    #i = 0
    #while (error > 0.02):
        for start, end in zip(range(0, len(trXY), bach), range(bach, len(trXY)+1, bach)):
            sess.run(train_op, feed_dict={X: trXY[start:end], Y: np.vstack(trainZ[start:end])})
        #print("Test = ", np.vstack(testZ))
        #print("Predicted = ", sess.run(py_x, feed_dict={X: teXY}))
        #print(np.argmax(testZ, axis=0) == sess.run(py_x, feed_dict={X: teXY}))
        #error = np.mean(MSE(np.vstack(testZ),sess.run(py_x, feed_dict={X: teXY})))
        predicted = sess.run(py_x, feed_dict={X: teXY})
        error = MSE(np.vstack(testZ),predicted)
        print(i, error)
        #print(i, np.mean(np.argmax(np.vstack(testZ), axis=0) == sess.run(predict_op, feed_dict={X: teXY})))
        i+=1
    saver.save(sess,"mlp/session.ckpt")