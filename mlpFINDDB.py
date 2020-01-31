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

solvers = [ "ConvAsm3x3U",
			"ConvAsm1x1U",
			"ConvAsm1x1UV2",
			"ConvBiasActivAsm1x1U",
			"ConvAsm5x10u2v2f1",
			"ConvAsm5x10u2v2b1",
			"ConvAsm7x7c3h224w224k64u2v2p3q3f1",
			"ConvOclDirectFwd11x11",
			"ConvOclDirectFwdGen",
			"ConvOclDirectFwd3x3",
			"ConvOclDirectFwd",
			"ConvOclDirectFwdFused",
			"ConvOclDirectFwd1x1",
			"ConvBinWinograd3x3U",
			"ConvBinWinogradRxS",
			"ConvAsmBwdWrW3x3",
			"ConvAsmBwdWrW1x1",
			"ConvOclBwdWrW2<1>",
			"ConvOclBwdWrW2<2>",
			"ConvOclBwdWrW2<4>",
			"ConvOclBwdWrW2<8>",
			"ConvOclBwdWrW2<16>",
			"ConvOclBwdWrW2NonTunable",
			"ConvOclBwdWrW53",
			"ConvOclBwdWrW1x1",
			"ConvHipImplicitGemmV4R1Fwd",
			"ConvHipImplicitGemmV4Fwd",
			"ConvHipImplicitGemmV4_1x1",
			"ConvHipImplicitGemmV4R4FwdXdlops",
			"ConvHipImplicitGemmV4R4Xdlops_1x1",
			"ConvHipImplicitGemmV4R1WrW",
			"ConvHipImplicitGemmV4WrW",
			"gemm",
			"fft",
			"ConvWinograd3x3MultipassWrW<3, 4>",
			"ConvBinWinogradRxSf3x2",
			"ConvWinograd3x3MultipassWrW<3, 5>",
			"ConvWinograd3x3MultipassWrW<3, 6>",
			"ConvWinograd3x3MultipassWrW<3, 2>",
			"ConvWinograd3x3MultipassWrW<3, 3>",
			"ConvWinograd3x3MultipassWrW<7, 2>",
			"ConvWinograd3x3MultipassWrW<7, 3>",
			"ConvWinograd3x3MultipassWrW<7, 2, 1, 1>",
			"ConvWinograd3x3MultipassWrW<7, 3, 1, 1>",
			"ConvWinograd3x3MultipassWrW<1, 1, 7, 2>",
			"ConvWinograd3x3MultipassWrW<1, 1, 7, 3>",
			"ConvWinograd3x3MultipassWrW<5, 3>",
			"ConvWinograd3x3MultipassWrW<5, 4>",
			"ConvHipImplicitGemmV4R4WrWXdlops",
			"ConvHipImplicitGemmV4R4GenFwdXdlops",
			"ConvHipImplicitGemmV4R4GenWrWXdlops", 
			"ConvBinWinogradRxSf2x3"]


feature_names=[ "C","Hin", "Win", "x", "y", "K", "Hout", "Wout", "n", "padH", "padW", "strideH", "strideW", "dilationH", "dilationW", "direction", "group" ]



strargs = sys.argv
data = genfromtxt(strargs[1], delimiter=',') # Input data X
labels = genfromtxt(strargs[2], delimiter=',') # Output labels Y


xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, random_state=52)
#Convert data into 3d tensor
xtrain3d = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1], 1))
xtest3d = np.reshape(xtest,(xtest.shape[0],xtest.shape[1], 1))

batches=1024
inittype='glorot_normal'
droprate=0.2
pat=3
label_name="solver"

def create_cnn_model(inittype):

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=1, input_shape=(len(feature_names), 1), kernel_initializer=inittype, name='conv1_128_1filter'))
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

	model.add(tf.keras.layers.Dense(1024, kernel_initializer=inittype)) 
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Dropout(droprate))

	model.add(tf.keras.layers.Dense(4096, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Dropout(droprate))

	model.add(tf.keras.layers.Dense(4096, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Dropout(droprate))

	model.add(tf.keras.layers.Dense(1024, kernel_initializer=inittype))
	model.add(tf.keras.layers.BatchNormalization(trainable=True))
	model.add(tf.keras.layers.ReLU())

	model.add(tf.keras.layers.Softmax())

	return model

cnnModel = create_cnn_model(inittype)

exit_acc_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=pat)
exit_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=pat)

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cnnModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = cnnModel.fit(xtrain3d, ytrain, batch_size=batches, epochs=1000, callbacks=[exit_acc_callback, exit_loss_callback, tensorboard_callback], validation_split=0.15, validation_freq=4, verbose=2)


print('\n# Evaluating on the test data.')
results = cnnModel.evaluate(xtest3d, ytest, verbose=0, batch_size=batches)
print('loss, acc: ', results)





