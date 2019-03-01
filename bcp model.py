# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:39:21 2019

@author: uvansankar
"""

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('H:\Medium\Machine Learning Model For Breast Cancer Prediction\data.csv')

X = dataset.drop('diagnosis',axis=1).values
Y = dataset['diagnosis'].values
dataset.head()

print("Cancer Data Dimensions : {}".format(dataset.shape))

#print(dataset.groupby('diagnosis').size())
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
#print(Y_pred)
#print(Y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred)*100)

