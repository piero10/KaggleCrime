import csv
import numpy as np
import pandas
import random
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

import json
import math
import SupportFunctions as sp
import scipy.stats as stats

class DataStorage:
    TrainData = []
    TestData = []
    Labels = []
    LabelsN = []

    TrainDataHalf1 = []
    TrainDataHalf2 = []

    LabelsDataHalf1 = []
    LabelsDataHalf2 = []

    LabelsNHalf1 = []
    LabelsNHalf2 = []
    
    commonPath = "C:\\python_kaggle\\crime_spyder_project\\"

    FeaturesNames = []

    allPosibleLabels = {"ARSON" : 0, 
                        "ASSAULT" : 1,
                        "BAD CHECKS" : 2,
                        "BRIBERY" : 3,
                        "BURGLARY" : 4,
                        "DISORDERLY CONDUCT" : 5,
                        "DRIVING UNDER THE INFLUENCE" : 6,
                        "DRUG/NARCOTIC" : 7, 
                        "DRUNKENNESS" : 8,
                        "EMBEZZLEMENT" : 9,
                        "EXTORTION" : 10, 
                        "FAMILY OFFENSES" : 11,
                        "FORGERY/COUNTERFEITING" : 12, 
                        "FRAUD" : 13,
                        "GAMBLING" : 14,
                        "KIDNAPPING" : 15,
                        "LARCENY/THEFT" : 16,
                        "LIQUOR LAWS" : 17,
                        "LOITERING" : 18,
                        "MISSING PERSON" : 19,
                        "NON-CRIMINAL" : 20,
                        "OTHER OFFENSES" : 21, 
                        "PORNOGRAPHY/OBSCENE MAT" : 22, 
                        "PROSTITUTION" : 23,
                        "RECOVERED VEHICLE" :24,
                        "ROBBERY" : 25, 
                        "RUNAWAY" : 26, 
                        "SECONDARY CODES" : 27,
                        "SEX OFFENSES FORCIBLE" : 28,
                        "SEX OFFENSES NON FORCIBLE" : 29,
                        "STOLEN PROPERTY" : 30,
                        "SUICIDE" : 31,
                        "SUSPICIOUS OCC" : 32,
                        "TREA" : 33,
                        "TRESPASS" : 34,
                        "VANDALISM" : 35, 
                        "VEHICLE THEFT" : 36, 
                        "WARRANTS" : 37, 
                        "WEAPON LAWS" : 38}

    allPosibleLabelsCount = 1



    def __init__ (self):
        self.allPosibleLabelsCount = len(self.allPosibleLabels)



    def GetDataForFeature(self):
        return self.TrainData, self.TestData



    def GetLabels(self, ind):
        return self.Labels[:,ind]



    def GetLabelsHalf(self, ind, half):
        labels = self.Labels[:,ind]
        return labels[half - 1:len(labels):2]



    def LoadDataFromFile(self):
        dataTogether = np.load(self.commonPath + "dump.npy")
        print(dataTogether)
    
             

    def LabelsToNums(self, labels):
        res = np.zeros((len(labels), 39))
        
        for i in range(0,len(labels)):
            if labels[i] in self.allPosibleLabels.keys():
                labelNum = self.allPosibleLabels[labels[i]]
                res[i, labelNum] = 1

        return res



    def ScaleData(self):
        scaler = StandardScaler()
        self.TrainData = scaler.fit_transform(self.TrainData)
        self.TestData = scaler.transform(self.TestData)

        self.TrainDataHalf1 = self.TrainData[::2]
        self.TrainDataHalf2 = self.TrainData[1::2]



    def LoadData(self, fromBinaryFile = True, filtering = False,
                 UseExternalGeoDataNearestest = True,  UseExternalGeoDataPlaces = True):
        if fromBinaryFile:
            dataTogether = np.load(self.commonPath + "dump_allData.npy")

            self.Labels = np.load(self.commonPath + "dump_labels.npy")
            self.LabelsN =  sp.LabelsSimpleFromLabels(self.Labels)

            self.TrainData = dataTogether[0:len(self.Labels)]
            self.TestData = dataTogether[len(self.Labels):len(dataTogether)]

            print("data loaded from file." )            
        else:
            trainDataRaw, labels = self.LoadTrainDataAndLabels() #(firstRows = 5000)

            testDataRaw = self.LoadTestData() #(firstRows = 5000)

            self.Labels = self.LabelsToNums(labels[:, 0])
            self.LabelsN = sp.LabelsSimpleFromLabels(self.Labels)

            dataTogether = np.concatenate((trainDataRaw, testDataRaw))
            dataTogether = self.DataNormalization(dataTogether) 
            print("data normilized." ) 

            np.save(self.commonPath + "dump_allData", dataTogether)
            np.save(self.commonPath + "dump_labels", self.Labels)

            self.TrainData = dataTogether[0:len(trainDataRaw)]
            self.TestData  = dataTogether[len(trainDataRaw): len(dataTogether)]

            print("data saved to file." ) 



        if UseExternalGeoDataNearestest:
            nearestData = self.LoadGeoNearestData()
            self.TrainData = np.c_[self.TrainData, nearestData[0:len(self.Labels)]]
            self.TestData = np.c_[self.TestData, nearestData[len(self.Labels):len(dataTogether)]]      

        if UseExternalGeoDataPlaces:
            self.GeoData = self.GeoDataPlaces()
            self.TrainData = np.c_[self.TrainData, self.GeoData[0:len(self.Labels)]]
            self.TestData = np.c_[self.TestData, self.GeoData[len(self.Labels):len(dataTogether)]]


        '''self.TrainDataHalf1 = self.TrainData[::2]
        self.TrainDataHalf2 = self.TrainData[1::2]

        self.LabelsDataHalf1 = self.Labels[::2]
        self.LabelsDataHalf2 = self.Labels[1::2]

        self.LabelsNHalf1 = self.LabelsN[::2]
        self.LabelsNHalf2 = self.LabelsN[1::2]'''
		
		self.TrainDataHalf1, self.TrainDataHalf2, self.LabelsDataHalf1, self.LabelsDataHalf2, self.LabelsNHalf1, self.LabelsNHalf2 = 
		sp.SplitTrainDataByWeeks(self.TrainData, self.Labels, self.LabelsN)
		

        if filtering:
            self.TrainDataHalf1,  self.LabelsDataHalf1 = self.DeleteFromTrainDataAllRowsWithCorrelationLessThan(self.TrainDataHalf1,  self.LabelsDataHalf1, 402, 0.99)


			

    def LoadGeoNearestData(self):
        dat = np.load(self.commonPath + "dump_allData_nearest.npy")
        
        col_mean = stats.nanmean(dat[:,1])
        dat = dat[:,1]
        dat[dat != dat] = col_mean
        return dat



    def GeoDataPlaces(self, delNotUsableFeatures = True, UseSumColumn = True):
        dat = np.load(self.commonPath + "dump_allData_geoInfo.npy")
        #status = dat[:,0]
        dat = np.delete(dat, 0, 1) # remove first "status" column
        sm = sum(sum(dat))
        #
        if delNotUsableFeatures:
            dat = dat[:,dat.sum(axis=0) > sm * 0.0001]

        return dat



    def DateTimeNormalization(self, column):

        res = [0] * len(column)

        firstDate = datetime.strptime(column[0], "%Y-%m-%d %H:%M:%S").date() - timedelta(days=1)

        for i in range(0, len(column)):
            dt = datetime.strptime(column[i], "%Y-%m-%d %H:%M:%S")
            time = (dt - timedelta(hours=6)).time()
            
            t = (time.hour * 60 + time.minute)

            date = dt.date()
            
            daysDelta = firstDate - date + timedelta(days=2)
            day_of_year = dt.timetuple().tm_yday
            mn = t % 60
            roundTimeMin = 0
            if mn in [0, 15, 30, 45]:
                roundTimeMin = 1

            res[i] = [daysDelta.days, t, day_of_year, roundTimeMin] 

        return res
     

         
    def DataNormalization(self, data):
        #streets = sp.Streets(data[:, 3])
        corner = sp.IfCorner(data[:, 3])
        coordinates = sp.Coordinates(data[:, 4:6])
        severalActionsInSameTime = sp.SeveralActionsInSameTime(coordinates, data[:, 0])
        streetsTypes = sp.StreetType(data[:, 3]) 
        dateTime = self.DateTimeNormalization(data[:, 0])
        distriction = sp.PDDistriction(data[:, 2])
        weekDaysBinarized = sp.WeekDaysBinarization(data[:, 1])
        
        dataNormalized = np.c_[dateTime, dataNormalized, weekDaysBinarized, coordinates, distriction, streetsTypes, severalActionsInSameTime, corner]

        return dataNormalized



    def LoadTrainDataAndLabels(self, firstRows = 0):
        myfile = self.commonPath + "train.csv"
        df = pandas.read_csv(myfile)

        a = df.values[:, 0:1]
        b = df.values[:, 3:5]
        c = df.values[:, 6:9]

        trainData = np.c_[a, b]
        trainData = np.c_[trainData, c]
        labels = np.c_[df.values[:, 1:3], df.values[:, 5]]

        if firstRows > 0:
            trainData = trainData[0:firstRows]
            labels = labels[0:firstRows]

        # delete all null coordinates
        i = 0
        while i < len(trainData):
            if trainData[i][4] != trainData[i][4]:
                 trainData = np.delete(trainData, (i), axis = 0) 
                 labels = np.delete(labels, (i), axis = 0) 

            i = i + 1

        return (trainData, labels)



    def LoadTestData(self, firstRows = 0):
        myfile =  self.commonPath + "test.csv"
        df = pandas.read_csv(myfile)
        testData = df.values[:, 1:7]

        if firstRows > 0:
            testData = testData[0:firstRows]
                             
        return testData



    def LoadSubmission(self):
        myfile =  self.commonPath + "su.csv"
        df = pandas.read_csv(myfile, delimiter = ';')
                             
        return df.values[:, 1:7]


    def LoadAnsewer(self, num): 
        myfile = self.commonPath + "my_submission_" + str(num) +".csv"
        df = pandas.read_csv(myfile, delimiter = ';', dtype=np.float32)
        res = df.values[:, 1:40]
        return res



    def SaveResult(self, result):
        myfile = self.commonPath + "diffBetweenLabelAndMyAnswer.csv"
        b = open(myfile, 'w', newline='')
        f = csv.writer(b)

        for i in range(len(result)):
            t = str(result[i])
            f.writerow(t)            

        print("evaluation saved")
        b.close()



    def DeleteFromTrainDataAllRowsWithCorrelationLessThan(self, trainData, labels, num, threadhold = 0.5):
        result = self.LoadAnsewer(num)
        
        ha = set([2, 3, 10, 11, 14, 22, 29, 31, 33])

        trainDataNew = [0] * len(trainData)
        labelsNew = [0] * len(trainData)

        j = 0
        for i in range(0, len(trainData)):
            d = abs(labels[i] - result[i])
            sm = sum(d)

            #s = labels[i][2] + labels[i][3] + labels[i][10] + labels[i][11] + labels[i][14] + labels[i][22] + labels[i][29] + labels[i][31] + labels[i][33]
            ssss = labels[i][1] + labels[i][16] + labels[i][20] + labels[i][21] + labels[i][32] + labels[i][35]
            #if sm < 2 * 0.95 and ssss < 0.999:  
            if  ssss < 0.999 or sm < 2 * 0.95: 
                trainDataNew[j] = trainData[i]
                labelsNew[j] = labels[i]
                j = j + 1


        trainDataNew = trainDataNew[0:j]
        labelsNew = labelsNew[0:j]

        r =[0] * 39
        l = np.array(labelsNew).reshape((len(labelsNew), len(labelsNew[0])))

        for i in range(0, 39):
            r[i] = sum(l[:,i])

        return (trainDataNew, l)



    def Save(self, data):
        a = random.randint(1, 1000)   
        myfile = self.commonPath + "my_submission1_" + str(a) +".csv"

        b = open(myfile, 'wb')
        f = csv.writer(b)

        ff = []
        for key, value in self.allPosibleLabels.items():
            ff.append(key)

        ff.sort()
        ff.insert(0, "Id")

        f.writerow(ff)

        for i in range(len(data)):
            t = [""] * (len(data[i]) + 1)
            t[0] = str(i)

            for ii in range(len(data[i])):
                t[ii + 1] = str(round(data[i, ii], 5)) 

            f.writerow(t)            

        print("submission saved")
        b.close()


