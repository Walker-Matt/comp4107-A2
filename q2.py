import tensorflow as tf
import numpy as np
import random

input_data = open("q2-patterns.txt", 'r')
lines = input_data.readlines()

trX = np.zeros((31,35), dtype = int)
trY = np.identity(31)

lineNum = 0
for i in range(31):
    for j in range(7):
        for k in range(5):
            trX[i][(j*5)+k] = lines[lineNum][k]
        lineNum = lineNum + 1
        
def distort(level):
    matrix = np.zeros((31,35), dtype = bool)
    for i in range(31):
        matrix[i] = noisy(trX[i], level)
    return matrix
        
def noisy(grid, level):
    for i in range(level):
        x = random.randint(0, 34)
        grid[x] = flip(grid[x])
    return grid
        
def flip(val):
    if(val == True):
        return False
    else:
        return True
    
teX = distort(3)
teY = trY
    
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

size_h = tf.constant(5, dtype=tf.int32)

X = tf.placeholder("float", [None, 35])
Y = tf.placeholder("float", [None, 31])

w_h = init_weights([35, size_h])
w_o = init_weights([size_h, 31])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
#predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(range(0,len(trX[0]),1))
    for i in range(1):
        for start, end in zip(range(0, len(trX), 1), range(1, len(trX)+1, 1)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        predicted = sess.run(py_x, feed_dict={X: teX})
        print(predicted)
    saver.save(sess,"mlp/session.ckpt")

