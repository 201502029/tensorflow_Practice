import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]

W = tf.placeholder(tf.float32)
hypothesis = W*x
cost = tf.reduce_mean(tf.square(hypothesis - y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()



x_data = [1,2,3]
y_data = [1,2,3]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = W*x

cost = tf.reduce_mean(tf.square(hypothesis - y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * x - y) * x)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={x: x_data, y: y_data})
    print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run(W))