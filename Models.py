from sklearn import svm
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model

import numpy as np

import sys
import os
import time
import theano
import theano.tensor as T
import random
import lasagne


def PredictLinearLasso(trainData, labels, testData):
    clf = linear_model.Lasso(alpha = 0.1)
    clf.fit(trainData, labels)
    predict = clf.predict(testData)
    return predict


'''
def build_mlp(input_var=None, input_dim = 1):
    l_in = lasagne.layers.InputLayer(shape = (None, input_dim), 
                                     input_var = input_var)

    l_out = lasagne.layers.DenseLayer(l_in, num_units=1,
                                   nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_out


def Lasagne(trainData, labels, testData):
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    network = build_mlp(input_var, trainData.shape[1])
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    labels.shape = (len(labels), 1)
    trainData = trainData[0:1000]
    labels = labels[0:1000]

    trainData = trainData * 1000
    trainData = trainData.astype(int)
    labels = labels.astype(int)

    testData = testData[0:1000]
    testData = testData * 1000
    testData = testData.astype(int)


    num = 2
    for epoch in range(num):
        train_err = train_fn(trainData, labels)
        print(" iter: " + str(epoch) + "    training loss: " + str(train_err))

    for i in range(len(testData)):
        err, acc = val_fn([testData[i]], [[0]]) 
        print(i, err, acc)
'''


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def build_mlp(input_var=None, input_dim = 1):
    l_in = lasagne.layers.InputLayer(shape=(None, input_dim),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units= int(input_dim / 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    l_out = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=39,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out



def Lasagne(trainData, labels, testData):
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    network = build_mlp(input_var, trainData.shape[1])
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)

    #labels.shape = (len(labels), 1)
    tr = 100000
    trainData = trainData[0:tr] 
    testData = testData[0:1000]
    labels = labels[0:tr]

    testData = testData * 100000 + 1
    trainData = trainData * 100000 + 1

    labels = labels * 10000 + 1

    trainData = trainData.astype(int)
    labels = labels.astype(int)
    testData = testData.astype(int)


    num = 10
    for epoch in range(num):
        train_err = 0
        train_batches = 0

        for batch in iterate_minibatches(trainData, labels, 1000, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        print(" iter: " + str(epoch) + "    training loss: " + str(train_err / train_batches))

    #for i in range(len(testData)):
    #nullres = np.zeros((len(testData), 39), dtype = int)
    #b = nullres[0:500]
    #err = val_fn(a, b) 

    predict_function = theano.function([input_var], prediction)
    res = predict_function(testData)

    return res



def PredictGrad(trainData, labels, testData, est = 100, max_dep = 5, min_samples_spl = 1):
    clf = GradientBoostingClassifier(n_estimators = est, max_depth = max_dep, min_samples_split = min_samples_spl)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict[:,1]



def PredictAda(trainData, labels, testData, est = 100):
    clf = AdaBoostClassifier(n_estimators=est)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict[:,1]



def PredictGauss(trainData, labels, testData):
    clf = GaussianNB()
    clf.fit(trainData, labels)
    predict = clf.predict(testData)
    #predict1 = clf.predict_proba(testData)
    return predict



def PredictRandomForest(trainData, labels, testData, dep = 20, n_est = 100, max_feat = 5):
    clf = RandomForestClassifier(max_depth = dep, n_estimators = n_est, max_features = max_feat)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict[:,1] 


# ????? ?????
def PredictSVC(trainData, labels, testData):
    clf = SVC(gamma=0.2)
    clf.fit(trainData, labels)
    predict = clf.predict_proba(testData)
    return predict
