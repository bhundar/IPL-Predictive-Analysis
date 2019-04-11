import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Ecnode variables such as batsmen and bowlers
def playerEncoding(players):
	encoder = LabelEncoder()
	encoder.fit(players)
	return encoder.transform(players)

def dismissalEncoding(rows, dismissals):
	r = np.zeros((rows, 2))
	for i, row in enumerate(dismissals):
		if row is not np.nan:
			r[i, 1] = 1
		else:
			r[i, 0] = 1
	return r

def encode():
	dataset = pd.read_csv("/Users/bhundar/Downloads/IPLPredictor/Data/deliveries.csv")
	innings = dataset[dataset.columns[1]].values
	over = dataset[dataset.columns[4]].values
	ball = dataset[dataset.columns[5]].values
	batsmen = playerEncoding(dataset[dataset.columns[6]].values)
	bowlers = playerEncoding(dataset[dataset.columns[8]].values)
	playersDismissed = dataset[dataset.columns[18]].values
	playersDismissed = dismissalEncoding(len(playersDismissed), playersDismissed)
	X = np.column_stack((innings, over, ball, batsmen, bowlers))
	return (X, playersDismissed)

X, Y = encode()

X, Y = shuffle(X, Y, random_state=10)

train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=415)

print("Dimensions: ")
print(train_x.shape, train_y.shape)

# ML Algorithm Parameters
learingRate = 0.01
trainingEpochs = 100
errorHistory = np.empty(shape=[1], dtype=float)
nDim = X.shape[1]
print("nDim", nDim)
nClass = Y.shape[1]

# Number of Neurons for each Hidden Layers
neuronHidden1 = 60
neuronHidden2 = 60
neuronHidden3 = 60
neuronHidden4 = 60

# n x nDim Matrix
x = tf.placeholder(tf.float32, [None, nDim]) 
W = tf.Variable(tf.zeros([nDim, nClass]))
b = tf.Variable(tf.zeros([nClass]))
y_ = tf.placeholder(tf.float32, [None, nClass])

# Define the model
def multilayerPerceptronModelling(x, weights, biases):
	# Hidden
	SigmoidFuncLayer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	SigmoidFuncLayer = tf.nn.sigmoid(SigmoidFuncLayer)
	# Hidden
    	SigmoidFuncLayer2 = tf.add(tf.matmul(SigmoidFuncLayer, weights['h2']), biases['b2'])
	SigmoidFuncLayer2 = tf.nn.sigmoid(SigmoidFuncLayer2)
	# Hidden
	SigmoidFuncLayer3 = tf.add(tf.matmul(SigmoidFuncLayer2, weights['h3']), biases['b3'])
	SigmoidFuncLayer3 = tf.nn.sigmoid(SigmoidFuncLayer3)
	# Hidden
	ReLULayer = tf.add(tf.matmul(SigmoidFuncLayer3, weights['h4']), biases['b4'])
	ReLULayer = tf.nn.relu(ReLULayer)
	# Output 
	LinearLayer = tf.matmul(ReLULayer, weights['out']) + biases['out']
	return LinearLayer
    
# Define weights and biases for each layer
weights = {
            'h1': tf.Variable(tf.truncated_normal([nDim, neuronHidden1])),
            'h2': tf.Variable(tf.truncated_normal([neuronHidden1, neuronHidden2])),
            'h3': tf.Variable(tf.truncated_normal([neuronHidden2, neuronHidden3])),
            'h4': tf.Variable(tf.truncated_normal([neuronHidden3, neuronHidden4])),
            'out': tf.Variable(tf.truncated_normal([neuronHidden4, nClass]))
        }

biases = {
            'b1': tf.Variable(tf.truncated_normal([neuronHidden1])),
            'b2': tf.Variable(tf.truncated_normal([neuronHidden2])),
            'b3': tf.Variable(tf.truncated_normal([neuronHidden3])),
            'b4': tf.Variable(tf.truncated_normal([neuronHidden4])),
            'out': tf.Variable(tf.truncated_normal([nClass]))
        }

# Initialize all variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = multilayerPerceptronModelling(x, weights, biases)

# Define error function and optimizer
errorFunc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learingRate).minimize(errorFunc)

session = tf.Session()
session.run(init)
PATH = os.getcwd() + "/models/ipl.ckpt"

# Calculate the error for each epoch
meanSqrErrHistory = []

print("Training: ")

for epoch in range(trainingEpochs):
	session.run(training_step, feed_dict={x: train_x, y_: train_y})
	error = session.run(errorFunc, feed_dict={x: train_x, y_: train_y})
	errorHistory = np.append(errorHistory, error)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	pred_y = session.run(y, feed_dict={x: test_x})
	meanSqrErr = tf.reduce_mean(tf.square(pred_y - test_y))
	meanSqrErr_ = session.run(meanSqrErr)
	meanSqrErrHistory.append(meanSqrErr_)
	accuracy = (session.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    
	if epoch % 15 == 0:
		print('Epoch: ', epoch, ' Error: ', error, ' Mean Square Error: ', meanSqrErr_, " Train Accuracy: ", accuracy)
 
modelPath = saver.save(session, PATH, global_step=trainingEpochs)
print("Model Path: %s" % modelPath)

# Print final accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test accuracy: ", (session.run(accuracy, feed_dict={x: test_x, y_:test_y})))

# Final Mean Square Error
pred_y = session.run(y, feed_dict={x: test_x})
meanSqrErr = tf.reduce_mean(tf.square(pred_y - test_y))
print("Mean Square Error: %.4f" % session.run(meanSqrErr))