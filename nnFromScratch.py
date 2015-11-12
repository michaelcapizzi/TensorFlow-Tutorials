__author__ = 'mcapizzi'

import tensorflow as tf
import math

#https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/mnist.py

###############
#INFERENCE
###############

#perceptron
def perceptron(inputX, w, activationFunction, b=None):
    #don't know if the naming and scope is done correctly
        #http://www.tensorflow.org/how_tos/graph_viz/index.md
    with tf.name_scope("perceptron") as scope:
        inputX = inputX
        w = w
        b = b

        if b is None:
            into = tf.matmul(inputX, w, name="into")
        else:
            # into = tf.add(
            #     tf.matmul(inputX, w),
            #     b)
            into = tf.nn_ops.bias_add(
                                        tf.matmul(inputX, w, name="into"),
                                        b,
                                        name="into + bias")

    if activationFunction == "sigmoid":
        activate = tf.nn.sigmoid(into, name="activation")
    else:
        activate = tf.nn.tanh(into, name="activation")

    return activate


#single layer neural network
def buildMLP1(inputX, w_In, w_Out, activationFunction, b_In=None, b_Out=None):
    #don't know if the naming and scope is done correctly
        #http://www.tensorflow.org/how_tos/graph_viz/index.md
    with tf.name_scope("hidden") as scope:
        inputX = inputX
        w_In = w_In
        b_In = b_In

        #W_in * input + b_In
        if b_In is None:
            into = tf.matmul(inputX, w_In, name="into")
        else:
            # into = tf.add(
            #     tf.matmul(inputX, w_In),
            #     b_In)
            into = tf.nn_ops.bias_add(
                                        tf.matmul(inputX, w_In, name="into"),
                                        b_In,
                                        name="into + bias")

        #activation
        if activationFunction == "sigmoid":
            activate = tf.nn.sigmoid(into, name="activation")
        else:
            activate = tf.nn.tanh(into, name="activation")

    #W_out * activate + b_Out
    if b_Out is None:
        out = tf.matmul(activate, w_Out, name="out")
    else:
        # out = tf.add(
        #     tf.matmul(activate, w_Out),
        #     b_Out)
        out = tf.nn_ops.bias_add(
                                    tf.matmul(activate, w_Out, name="out"),
                                    b_Out,
                                    name="out + bias")

    return out


#TODO generalize method to build arbitrary number of hidden layers

###############
#LOSS
###############

def calculateLoss(logits, labels):
    #calculate cross entropy
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="cross-entropy")

    #calculate loss
    loss = tf.reduce_mean(crossEntropy, name="loss")

    return loss


###############
#TRAINING
###############

def training(loss, learningRate):
    #scalar summary for snapshot of loss
    tf.scalar_summary(loss.op.name, loss)       #don't know what this does?

    #gradient optimizer
    optimizer = tf.train.GradientDescentOptimizer(learningRate)

    #global step variable
    globalStep = tf.Variable(0, name="global step", trainable=False)    #don't know what this does?

    #train_op can be passed to sess.run() to train
    trainOp = optimizer.minimize(loss, global_step=globalStep)

    return trainOp


###############
#EVALUATION
###############

#takes same inputs as calculateLoss?
def evaluate(logits, labels):
    #calculates how many of logits in top 1 matched label
    correct = tf.nn.in_top_k(logits, labels, 1)

    #returns tensor of boolean values
    return correct


########################################################################

#TODO figure out how to run the parts now
#https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py

########################################################################

#set up placeholders
w_In_P = tf.placeholder("float")
w_Out_P = tf.placeholder("float")
b_In_P = tf.placeholder("float")
b_Out_P = tf.placeholder("float")

#randomly initialize weights
    #how to do with -root(6/d) to root(6/d)
def initialize_weights(rows, columns):
    return tf.Variable(tf.random_normal([rows, columns], stddev=math.sqrt(6/(rows*columns))))
    # return tf.Variable(tf.random_normal([rows, columns], stddev=0.01))


#fill feed_dict
#TODO build

