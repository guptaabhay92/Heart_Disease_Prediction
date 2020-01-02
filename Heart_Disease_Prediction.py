# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:33:46 2020

@author: Abhay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('dataset.csv')

df.isnull().values.any()

df.corr()
target_zero_count=len(df.loc[df['target']==0])
target_ones_count=len(df.loc[df['target']==1])

df=pd.get_dummies(df,columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

X=df.drop(['target'],axis=1)
y=df['target']

#dividing into training and test dataset

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.20, random_state=0) 


#fiding best k value
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_score=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn, X, y, cv=10)
    knn_score.append(score.mean())
   
    
plt.plot([k for k in range(1, 21)], knn_score, color = 'red')
for i in range(1,21):
    plt.text(i, knn_score[i-1], (i, knn_score[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


## 1st predictiopn using K-Nearest Neighbour
knn=KNeighborsClassifier(n_neighbors=13)
score=cross_val_score(knn, X, y, cv=10)
score.mean()
    

#2nd using RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
score_rfc=cross_val_score(rfc, X, y, cv=10)
score_rfc.mean()

#3rd using Decision Tree

from sklearn.tree import DecisionTreeClassifier
d_tree=DecisionTreeClassifier()
score_decision=cross_val_score(d_tree, X, y, cv=10)
score_decision.mean()







