import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

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
    matrix = np.zeros((31,35), dtype = int)
    for i in range(31):
        matrix[i] = noisy(trX[i], level)
    return matrix
        
def noisy(grid, level):
    for i in range(level):
        x = random.randint(0, 34)
        grid[x] = flip(grid[x])
    return grid
        
def flip(val):
    if(val == 1):
        return 0
    else:
        return 1
    
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)
    

e = 150
percentages1 = np.array([], dtype = "float")
percentages2 = np.array([], dtype = "float")

noise = np.array([0,1,2,3])
teY = trY

size_h = tf.constant(15, dtype=tf.int32)

X = tf.placeholder("float", [None, 35])
Y = tf.placeholder("float", [None, 31])

w_h = init_weights([35, size_h])
w_o = init_weights([size_h, 31])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer(0.05).minimize(cost)
#predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

batch = 3

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #print(range(0,len(trX[0]),batch))
    for i in range(e):
        for start, end in zip(range(0, len(trX), batch), range(batch, len(trX)+1, batch)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    for n in noise:
        teX = distort(n)
        predicted = sess.run(tf.nn.sigmoid(py_x), feed_dict={X: teX})
        correct = 0    
        for i in range(31):
            if (np.argmax(predicted[i]) == np.argmax(teY[i])):
                correct += 1
        percent = 100*(1-(correct/31))
        print("Percentages1 = ", percent, "%")
        percentages1 = np.append(percentages1, percent)
    saver.save(sess,"mlp/session.ckpt")
    sess.close()
    


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #print(range(0,len(trX[0]),batch))
    for i in range(e):
        for start, end in zip(range(0, len(trX), batch), range(batch, len(trX)+1, batch)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        for start, end in zip(range(0, len(teX), batch), range(batch, len(teX)+1, batch)):
            sess.run(train_op, feed_dict={X: teX[start:end], Y: trY[start:end]})
        for start, end in zip(range(0, len(trX), batch), range(batch, len(trX)+1, batch)):                
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    for n in noise:
        teX = distort(n)
        predicted = sess.run(tf.nn.sigmoid(py_x), feed_dict={X: teX})
        correct = 0    
        for i in range(31):
            if (np.argmax(predicted[i]) == np.argmax(teY[i])):
                correct += 1
        percent = 100*(1-(correct/31))
        print("Percentages2 = ", percent, "%")
        percentages2 = np.append(percentages2, percent)
    saver.save(sess,"mlp/session.ckpt")
    sess.close()

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
one, = plt.plot(noise,percentages1, label = "Trained Without Noise")
three, = plt.plot(noise,percentages2, label = "Trained With Noise")
plt.title("Percentage Error vs Noise Level")
plt.xlabel("Noise Level")
plt.ylabel("Recognition Percentage Error (%)")
plt.legend(handles = [one,three], loc = 2)
plt.grid()
plt.show()