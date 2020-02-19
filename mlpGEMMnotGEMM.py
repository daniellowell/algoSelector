from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import subprocess

import matplotlib.pyplot as plt

import tensorflow as tf
import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from numpy import genfromtxt


feature_names=[ "C", "Hin", "Win", "x", "y", "K", "Hout", "Wout", "n", "padH", "padW", "strideH", "strideW", "dilationH", "dilationW", "direction", "group" ]



strargs = sys.argv
data = genfromtxt(strargs[1], delimiter=',') # Input data X
labels = genfromtxt(strargs[2], delimiter=',') # Output labels Y

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, random_state=52)
#Convert data into 3d tensor
xtrain3d = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1], 1))
xtest3d = np.reshape(xtest,(xtest.shape[0],xtest.shape[1], 1))
print("input shape (0,1): %d, %d ",xtrain.shape[0], xtrain.shape[1])
print("number of features: %d ", len("feature_names"))
np.savetxt("binary_testindata.csv", xtest, delimiter=",")
np.savetxt("binary_testlbldata.csv", ytest, delimiter=",")


batches=1024
#inittype='lecun_normal' # 'glorot_normal'
inittype='he_normal'

droprate=0.2
pat=3
label_name="solver"

def create_mlp_model(inittype):

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=1, input_shape=(17, 1), kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=1,  kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Flatten())


	model.add(tf.keras.layers.Dense(1024, kernel_initializer=inittype))#, input_shape=(17,)))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Dense(4096, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

#	model.add(tf.keras.layers.Dense(4096, kernel_initializer=inittype))
#	model.add(tf.keras.layers.BatchNormalization(trainable=True))
#	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Dense(1024, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#	model.add(tf.keras.layers.Dense(outputlen, activation='softmax', kernel_initializer=inittype))

	return model

model = create_mlp_model(inittype)

exit_acc_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=pat)
exit_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=pat)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], learning_rate=0.0001)

history = model.fit(xtrain3d, ytrain, batch_size=batches, epochs=1000, callbacks=[exit_acc_callback, exit_loss_callback], validation_split=0.15, validation_freq=4, verbose=2)


print('\n# Evaluating on the test data.')
results = model.evaluate(xtest3d, ytest, verbose=0, batch_size=batches)
print('loss, acc: ', results)

if(strargs[3]!=""):
	model.save(strargs[3], save_format="h5", include_optimizer=False)



