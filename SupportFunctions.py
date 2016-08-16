from sklearn.preprocessing import label_binarize
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
import hashlib
import sys


weekDays = {"Sunday" : 0, 
            "Monday" : 1,
            "Tuesday": 2, 
            "Wednesday" : 3, 
            "Thursday" : 4, 
            "Friday" : 5, 
            "Saturday" : 6}

districtions = {"BAYVIEW" : 0, 
                "NORTHERN" : 1,
                "INGLESIDE": 2, 
                "TARAVAL" : 3, 
                "MISSION" : 4, 
                "TENDERLOIN" : 5, 
                "RICHMOND" : 6,
                "CENTRAL" : 7, 
                "PARK" : 8,
                "SOUTHERN" : 9}


streestTypesDict = {"AV" : 0, 
                "BL" : 1,                
                "CR" : 2,                
                "CT" : 3,                
                "DR" : 4,                
                "LN" : 5,                 
                "PZ" : 6,                
                "PL" : 7,                
                "RD" : 8,                
                "ST" : 9,                
                "TR" : 10,                
                "HWY": 11,                
                "HY" : 12,                
                "WY" : 13,                
                "WAY": 14}


def LabelsSimpleFromLabels(labels):
    resN = np.zeros(len(labels))
    for i in range(0,len(labels)):
        for j in range(0,len(labels[i])):
            if labels[i,j] == 1:
                resN[i] = j
                break

    return resN



def Streets(streetColumn):
    streets = np.empty((len(streetColumn), 2), dtype=object)
    streetsTogether = np.empty(len(streetColumn), dtype=object)

    for i in range(0, len(streetColumn)):
        street = [w for w in streetColumn[i].split("/")]
        
        streets[i, 0] = "None"
        streets[i, 1] = "None"

        if len(street) > 0:
            firstStreet = [st for st in street[0].split() if st.isupper() and len(st) > 2] 
            if len(firstStreet) > 0:
                streets[i, 0] = ''.join(firstStreet)

        if len(street) > 1:
            secondString = [st for st in street[1].split() if st.isupper() and len(st) > 2]
            if len(secondString) > 0:
                streets[i, 1] = ''.join(secondString)

        streetsTogether[i] = streets[i, 0] + " " + streets[i, 1]
        streetsTogether[i] = streetsTogether[i].lower()

    r1 = my_hashing(streetsTogether, 100)

    return r1



def my_hashing(features, N):
    x = [0] * N
    featuresBinarized = [0] * len(features)
    for i in range(len(features)):
        strings = features[i].split()
        featuresBinarized[i] = [0] * N
        for j in range(0, len(strings)):
            if strings[j] != "None":
                sha = abs(hash(strings[j]))
                featuresBinarized[i][sha % N] = 1

    return featuresBinarized


def IfCorner(streetColumn):
    corner = [0]* len(streetColumn)

    for i in range(0, len(streetColumn)):
        streetsInAdress = [w for w in streetColumn[i].upper().split("/")]
        corner = len(streetsInAdress) - 1

    return corner


def StreetType(streetColumn):
    streetsTypesBin = np.zeros((len(streetColumn), 15), dtype=int)

    for i in range(0, len(streetColumn)):
        #street = [w for w in streetColumn[i].upper().split("/")]

        tp = np.zeros(15, dtype=int)
        types = []
        if len(street) > 0:
            if len(street) > 0:
                types = [st for st in street[0].split() if st in streestTypesDict.keys()] # and len(st) > 2 and not st == 'hwy' and not st == 'way'] 

            if len(street) > 1:
                tp1 = [st for st in street[1].split() if st in streestTypesDict.keys()]
                types.extend(tp1)

            for j in types:
                tp[streestTypesDict[j]] = 1

        streetsTypesBin[i] = tp

    return streetsTypesBin



def SeveralActionsInSameTime(coord, time):
    severalActionsInSameTime = [0] * len(time)
    for i in range(1, len(time)):
        k = i - 1
        same = False
        while IsActionHappendsInSamePlaceAndTime(coord[i][1], coord[i][0], time[i], coord[k][1], coord[k][0], time[k]):
            same = True
            k = k - 1

        if same:
            for j in range(k + 1, i + 1):
                severalActionsInSameTime[j] = i - k - 1

    return severalActionsInSameTime





def IsActionHappendsInSamePlaceAndTime(coordX1, coordY1, time1, coordX2, coordY2, time2):
    if (abs(coordX1 - coordX2) < 0.00000001 and abs(coordY1 - coordY2) < 0.00000001 and time1 == time2):
        return True

    return False




def Coordinates(coordinates):
    a = 0.0
    b = 0.0
    a_count = 0
    b_count = 0

    coord = [0] * len(coordinates)

    for i in range(0, len(coordinates)):
        if coordinates[i, 0] == coordinates[i, 0]:
            a += coordinates[i, 0]
            a_count += 1

        if coordinates[i, 1] == coordinates[i, 1]:
            b += coordinates[i, 1]
            b_count += 1

    ave0 = a / a_count
    ave1 = b / b_count 

    for i in range(0, len(coordinates)):
        if coordinates[i, 0] != coordinates[i, 0]:
            coordinates[i, 0] = ave0

        if coordinates[i, 1] != coordinates[i, 1]:
            coordinates[i, 1] = ave1

    coordinatesCircle = [0] * len(coordinates)
    for i in range(0, len(coordinates)):
        dist = math.sqrt((coordinates[i, 0] - ave0) * (coordinates[i, 0] - ave0) + (coordinates[i, 1] - ave1) * (coordinates[i, 1] - ave1))
        m = coordinates[i, 1] - ave1
        m = max(m, 0.001)
        angle = math.atan((coordinates[i, 0] - ave0) / m)

        coordinatesCircle[i] = [dist, angle, dist * angle]

    coordinates = np.c_[coordinates, coordinatesCircle]

    return coordinates



def PDDistriction(column):
    resColumns = [0] * len(column)
    for i in range(0, len(column)):
        resColumns[i] = [0] * 10
            
        if column[i] in districtions.keys():
            key = districtions[column[i]]
            resColumns[i][key] = 1

    return resColumns



def IFCorner(adressColumn):
    res = [0] * len(adressColumn)

    for i in range(0, len(adressColumn)):
        if "/" in adressColumn:
            res[i] = 1

    return res



def LabelsBinarization(self, labels):
    la = label_binarize(labels, classes=self.allPosibleLabels)
    return la



def WeekDaysBinarization(column):
    column1 = [0] * len(column)
    for i in range(0, len(column)):
        r = 7
        if column[i] in weekDays.keys():
            r = weekDays[column[i]]

        column1[i] = r

    myset = set(column1)
    mm = list(myset)
    r1 = label_binarize(column1, classes=mm)
    r1 = r1[:,0:7]
    r1 = np.column_stack((r1, column1))

    weekDay = [0] * len(column1)

    for i in range(0, len(column1)):
        weekDay[i] = 0
        if column1[i] == 0 or column1[i] == 6:
            weekDay[i] = 1

    r1 = np.column_stack((r1, weekDay))

    return r1


 
def ToHoursDistribution(tm):
    res = [0] * 24

    hour = math.floor(tm / 60)
    mn = tm - hour * 60

    if hour > 0:
        hourBefore = hour - 1
    else:
        hourBefore = 23

    if hour < 23:
        hourAfter = hour + 1
    else:
        hourAfter = 0

    add = mn - 30

    res[hour] = 2 - np.abs(add) / 30
    res[hourAfter] = mn / 60 
    res[hourBefore] = 1 - res[hourAfter]

    return res