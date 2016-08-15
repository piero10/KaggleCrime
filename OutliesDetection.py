import Data
import Metrics
import numpy as np
import timeit
import algos
import threading
import Models
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.covariance import EllipticEnvelope
from multiprocessing import Queue, Process, Pipe
from datetime import datetime
from sklearn import svm



def RunAllFeatures(trainData, labels, testData, featureFrom = 0, featureTo = 39, ):
    casesNum = featureTo - featureFrom
    fullprediction = [0] * casesNum

    for i in range(featureFrom, featureTo):
        algonum = algos.featureAlgos[i]
        algo = algos.allPossibleAlgorithms[algonum[1]]

        pred = Models.PredictGrad(trainData, labels[:, i], testData, est = 100)

        fullprediction[i - featureFrom] = pred
        dt = datetime.now()
        print(str(dt.hour) + "." + str(dt.minute) + "." + str(dt.second) + "   model " + str(i) + " done.")

    fullprediction = np.array(fullprediction)
    #conn.send(fullprediction)
    
    return fullprediction



def startWith4Threads(trainData, labels, testData):
    rr1 = RunAllFeatures(rainData, labels, testData, 0, 39)
    '''parent_conn1, child_conn1 = Pipe()
    parent_conn2, child_conn2 = Pipe()
    parent_conn3, child_conn3 = Pipe()
    parent_conn4, child_conn4 = Pipe()

    t1 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn1, 0, 10,  ))
    t2 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn2, 10, 20, ))
    t3 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn3, 20, 30, ))
    t4 = threading.Thread(target=RunAllFeatures, args = (trainData, labels, testData, child_conn4, 30, 39, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    
    rr1 = parent_conn1.recv()
    rr2 = parent_conn2.recv()
    rr3 = parent_conn3.recv()
    rr4 = parent_conn4.recv()

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    
    rr1 = np.row_stack((rr1, rr2))
    rr3 = np.row_stack((rr3, rr4))
    rr1 = np.row_stack((rr1, rr3))
    '''
    res = np.transpose(rr1)

    Normalization(res)

    return res



def startWith4ThreadsWithProve(trainData, labels, testData, trueLabels):
    res = startWith4Threads(trainData, labels, testData)
    print("") 

    if (len(res),len(res[0])) == (len(trueLabels),len(trueLabels[0])):
        for i in range(0, len(res[0])):
            res1 = Metrics.QualityMetrics.LinearQuality(trueLabels[:,i], res[:,i])
            res2 = Metrics.QualityMetrics.MLogLoss(trueLabels[:,i], res[:,i])

            print("model: " + str(i) + " result: " + str(res1) + ",   " + str(res2))
            
        print("") 
        res1 = Metrics.QualityMetrics.LinearQuality(trueLabels, res)
        res2 = Metrics.QualityMetrics.MLogLoss(trueLabels, res)

        print("result full = " + str(res1) + ",   " + str(res2))

    return res



data = Data.DataStorage()

data.LoadData(fromBinaryFile = True, filtering = True)
#parent_conn1, child_conn1 = Pipe()
result = RunAllFeatures(data.TrainData, data.Labels, data.TrainData, 0, 39)
#result = startWith4ThreadsWithProve(data.TrainData, data.Labels, data.TrainData, data.Labels)

#data.SaveResult(result)
#data.Save(result)
