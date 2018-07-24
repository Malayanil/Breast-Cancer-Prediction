import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict


df = pd.read_csv('biopsy.csv')


# x = df['Class'].value_counts()
# print(x)

X = df[['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']]
# print(X[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X[0:5])

Y = df['Class'].values
# print(Y[0:9])

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
# print ('Train set:', X_train.shape,  Y_train.shape)
# print ('Test set:', X_test.shape,  Y_test.shape)


nCount = range(1,20,1)
training_accuracy = []
test_accuracy = []

for i  in nCount:

    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    # print(neigh)
    
    Yhat = neigh.predict(X_test)
    # print(Yhat[0:5])

    # Accuracy Evaluation
    training_accuracy.append(metrics.accuracy_score(Y_train, neigh.predict(X_train)))
    # print("Train set Accuracy: ",training_accuracy)
    # plt.plot(i,metrics.accuracy_score(Y_train, neigh.predict(X_train)), color = 'black')
    test_accuracy.append(metrics.accuracy_score(Y_test, Yhat))
    # print("Test set Accuracy: ", test_accuracy)
    
plt.plot(nCount,training_accuracy, label ='Training Accuracy')
plt.plot(nCount, test_accuracy, label ='Testing Accuracy')
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.title('Convergence of Train and Test Graph')
plt.legend()
# plt.show()

neigh = KNeighborsClassifier(n_neighbors = 9).fit(X_train,Y_train)
Yhat = neigh.predict(X_test)

# Accuracy Evaluation
print('\n')
print('Train set Accuracy: ', metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print('Test set Accuracy: ', metrics.accuracy_score(Y_test, Yhat))

# K-Fold Implementation

kf = KFold(3, True)
kf.get_n_splits(X)
# print(kf)

for train_index, test_index in kf.split(X):
    # print('Train: ', train_index, 'Test: ', test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

# Cross Validation

kNN = KNeighborsClassifier(n_neighbors = 9)
score = cross_val_score(kNN, X, Y, cv=7)
print('\n')
print('Cross Validation Mean Score: ',score.mean())
