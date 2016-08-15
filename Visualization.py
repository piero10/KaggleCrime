import Data
import Metrics
import numpy as np
import timeit
import algos
import threading
import Models
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel

from multiprocessing import Queue, Process, Pipe
from datetime import datetime
import pandas
import matplotlib.pyplot as plt
import seaborn as sns


data = Data.DataStorage()

data.LoadData(fromBinaryFile = True)

sns.set()

dat = np.column_stack((data.TrainData[:,0], data.TrainData[:,14:16]))
dat = np.column_stack((dat, data.TrainData[:,1]))
# [st for st in street[0].split() if st.isupper() and len(st) > 2] 
labels = np.array(data.GetLabelsInOneColumn())

dat = np.array(dat)

l = np.column_stack((dat, labels))

l = [w for w in l if w[1] < -122 and w[1] > -123 and w[0] < 5]
l = np.array(l)
l = l[:, 1:5]

df = pandas.DataFrame(l, columns=['x', 'y', 'time', 'label'])
sns.pairplot(df, hue = 'label')
sns.plt.show()

#plt.plot(coord[:,0], coord[:,1], 'ro')
#plt.axis([-122.5, -122.35, 37.7, 37.87])
#plt.show()
