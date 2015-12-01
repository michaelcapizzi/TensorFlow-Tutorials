import math

__author__ = 'mcapizzi'


from input_data import read_data_sets
import tensorflow as tf

#read in data
##################################################
mnist = read_data_sets("MNIST_data/", one_hot=True)

trainX, trainY, testX, testY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#build graph
###################################################
#autoencoder
#global parameter for number of hidden units
hiddenUnits = 25
inputUnits = 784
outputUnits = inputUnits
batchSize = 100             #size of batch to run through
alpha = .001                #learning rate


#set up placeholders (places where mnist data will appear)
input = tf.placeholder("float", name="Input", shape=[None, inputUnits])
label = tf.placeholder("float", name="Label")


# setup variables (parameters of the neural network model)
W1 = tf.Variable(tf.random_normal([inputUnits, hiddenUnits], mean=0, stddev=math.sqrt(float(6) / float(inputUnits * 2))), name="W1")
W2 = tf.Variable(tf.random_normal([hiddenUnits, outputUnits], mean=0, stddev=math.sqrt(float(6) / float(inputUnits * 2))), name="W2")
#with b1 being just a row
b1 = tf.Variable(tf.random_normal([1, hiddenUnits], mean=0, stddev=math.sqrt(float(6) / float(inputUnits * 2))), name="b1")
#with b1 being tiled
# b1 = tf.Variable(tf.random_normal([batchSize, hiddenUnits], mean=0, stddev=math.sqrt(float(6) / float(inputUnits * 2))), name="b1")
b2 = tf.Variable(tf.random_normal([1, outputUnits], mean=0, stddev=math.sqrt(float(6) / float(inputUnits * 2))), name="b2")
#with b2 being tiled
# b2 = tf.Variable(tf.random_normal([hiddenUnits, batchSize], mean=0, stddev=math.sqrt(float(6) / float(inputUnits * 2))), name="b2")


#feedforward graph
def feedForward(input_, W1_, W2_, b1_, b2, activation):
    #input to hidden
    z1 = tf.add (
                    tf.matmul   (
                                    input_,
                                    W1_
                                ),
                    b1_
                )

    #hidden activation
    if activation == "sigmoid":
        a1 = tf.nn.sigmoid(z1, name="Activation-sigmoid")
    elif activation == "tanh":
        a1 = tf.nn.tanh(z1, name="Activation-tanh")
    elif activation == "relu":
        a1 = tf.nn.relu(z1, name="Activation-relu")
    else:
        a1 = tf.nn.sigmoid(z1, name="Activation-sigmoid")

    #hidden to output
    z2 = tf.add (
                    tf.matmul   (
                                    a1,
                                    W2_
                                ),
                    b2
                )

    return z2


#output of the feed forward process
ffOp = feedForward(input, W1, W2, b1, b2, "relu")
# ffOp = feedForward(trainX, W1, W2, b1, b2, "relu")


#cost activation
#cross entropy portion
crossEntropyOp = tf.nn.softmax_cross_entropy_with_logits(ffOp, label, name="CrossEntropy")
#loss portion
lossOp = tf.reduce_mean(crossEntropyOp, name="Loss")


#gradient descent
gradientOp = tf.train.GradientDescentOptimizer(alpha).minimize(lossOp)


#for predictions
predictOp = tf.nn.softmax(ffOp, name="Predictions")


#variable initialization
initOp = tf.initialize_all_variables()

###################################################################

#open session
sess = tf.Session()


#initialize all variables
sess.run(initOp)


def train(numberOfIterations):
    #number of training items
    trainSize = trainX.shape[0]
    #items per batch
    batches = int(trainSize / batchSize)
    #initialize variable to be used later
    avgCost = 0

    #initial parameters (for comparison after training)
    initialW1 = W1.initialized_value()._AsTensor().eval(session=sess)
    initialW2 = W2.initialized_value()._AsTensor().eval(session=sess)
    initialb1 = b1.initialized_value()._AsTensor().eval(session=sess)
    initialb2 = b2.initialized_value()._AsTensor().eval(session=sess)


    #for each training iteration
    for i in range(numberOfIterations):
        #for each batch
        for j in range(batches):
            #TODO confirm this is correct
            trainBatchX = trainX[batchSize * j: batchSize * (j+1)]
            trainBatchY = trainY[batchSize * j: batchSize * (j+1)]
            #run training
                #need to use feed_dict to put data items into placeholder spots
            sess.run(gradientOp, feed_dict={input: trainBatchX, label: trainBatchY})
            #compute average cost for printing
            avgCost = sess.run(lossOp, feed_dict={input: trainBatchX, label: trainBatchY})
        #print
        print "iteration %s with average cost of %s" %(str(i),str(avgCost))






