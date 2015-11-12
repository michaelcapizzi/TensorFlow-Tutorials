__author__ = 'mcapizzi'

import tensorflow as tf
import math

#building models

#perceptron
def perceptron(inputX, w, activationFunction, b=None):
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




