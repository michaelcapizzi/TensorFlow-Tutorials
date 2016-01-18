__author__ = 'mcapizzi'

import numpy as np
import tensorflow as tf

#placeholders and run
#can set default dimensions for placeholder, or leave unnamed
spot1 = tf.placeholder("float64")
spot2 = tf.placeholder("float64")
spot3 = tf.placeholder("float64", shape=[None, None])   #don't need to type, but helps safeguard
add_op = tf.add(spot1, spot2)
add_op2 = tf.add(spot1, spot3)
sess= tf.Session()
output = sess.run(add_op, feed_dict={spot1: 1., spot2: [.1, .1]})
#if explicitly type, then need to make sure dimensions match
output2 = sess.run(add_op2, feed_dict={spot1: 1., spot3: np.array([[.1, .1]])})
output3 = sess.run(add_op2, feed_dict={spot1: 1., spot3: [[.1,.1]]})

#variables
#must initialize values
v1 = tf.Variable(np.zeros((5,5)))
init_op = tf.initialize_all_variables()
up = tf.assign(v1, v1 + 5)                                  #updates values of variable

#don't forget to initialize variables!
sess.run(init_op)

for i in range(3):
    print v1._AsTensor().eval(session=sess)
    out = sess.run(up)
    print type(out)
    print type(v1)

print v1._AsTensor().eval(session=sess)             #allows you to see the value of the variable

#built in neural network functions
#activation
tf.nn.sigmoid()
# tf.nn.tanh()
# tf.nn.relu()

#cost
#tf.nn.sigmoid_cross_entropy_with_logits
#tf.nn.softmax
#tf.nn.softmax_cross_entropy_with_logits

#BUILD FEEDFORWARD WITH NO ACTIVATION
#BUILD COST WITH tf.softmax_cross_entropy_with_logits(FF)
#BUILD PREDICT WITH tf.softmax(FF)


tf.nn.embedding_ops

