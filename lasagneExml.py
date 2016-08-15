from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import random

import lasagne

X_train = np.array([
    [1,2], 
    [4,1], 
    [2,3], 
    [1,1],
    [1,5],
    [6,0],
    [0,2],
    [4,2],
    [2,1],
    ]) #, [2, 3, 4, 0], [3, 1, 1, 0], [5, 6, 6, 0]])

y_train = np.array( [3, 5, 5, 2, 6, 6, 2, 6, 3])
y_train.shape = (9, 1)

X_test = np.array([
    [1,1], 
    [4,2], 
    [2,2], 
    [1,0],
    [1,4],
    [6,1],
    [0,1],
    [4,1],
    [2,3],
    ])

y_test = np.array([2, 6, 4, 1, 5, 7, 1, 5, 5])
y_test.shape = (9, 1)


def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape = (None, 2), 
                                     input_var = input_var)

    l_out = lasagne.layers.DenseLayer(l_in, num_units=1,
                                   nonlinearity=lasagne.nonlinearities.linear)

    return l_out


sz = 100
X_train = np.zeros((sz, 2), dtype = int)
y_train = np.zeros((sz), dtype = int)
X_test = np.zeros((sz, 2), dtype = int)
y_test = np.zeros((sz), dtype = int)

for i in range(sz):
    X_train[i] = [random.randint(-10, 10), random.randint(-10, 10)]
    y_train[i] = sum(X_train[i])
    
    X_test[i] = [random.randint(-10, 10), random.randint(-10, 10)]
    y_test[i] = sum(X_test[i])
    
y_train.shape = (sz, 1)    
y_test.shape = (sz, 1)



# Prepare Theano variables for inputs and targets
input_var = T.matrix('inputs')
target_var = T.matrix('targets')

network = build_mlp(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
#test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
#                                                        target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

num = 20
for epoch in range(num):
    train_err = train_fn(X_train, y_train)
    print(" iter: " + str(epoch) + "    training loss: " + str(train_err))

print(" ")
print(" ")
print(" ")
e = val_fn([[10,10]], [[40]])
print(e)

test_err = 0
for epoch in range(num):
    err, acc = val_fn(X_test, y_test)
    test_err += err

print(" ")
print("  validation loss:" + str(test_err / num))




