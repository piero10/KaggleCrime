import numpy as np

class QualityMetrics:

    @staticmethod
    def MLogLoss(actual, predicted, eps=1e-15):
        
        actual = np.array(actual)
        actual = actual.ravel()

        predicted = np.array(predicted)
        predicted = predicted.ravel()

        predicted[predicted < eps] = eps
        predicted[predicted > 1 - eps] = 1 - eps
        return (-1/actual.shape[0]*(sum(actual*np.log(predicted)))).mean()
   


    @staticmethod
    def LinearQuality(actual, predicted):
        
        actual = np.array(actual)
        actual = actual.ravel()

        predicted = np.array(predicted)
        predicted = predicted.ravel()

        res = abs(actual - predicted)
        return sum(res)


