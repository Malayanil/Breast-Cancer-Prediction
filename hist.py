import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('biopsy.csv')

X = df[['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class']]

X1 = X[['ClumpThickness']].values

plt.hist(X1)
plt.xlabel('ClumpThickeness')
plt.ylabel('Count')
plt.show()


X2 = X[['UniformityOfCellSize']].values

plt.hist(X2)
plt.xlabel('UniformityOfCellSize')
plt.ylabel('Count')
plt.show()


X3 = X[['UniformityOfCellShape']].values

plt.hist(X3)
plt.xlabel('UniformityOfCellShape')
plt.ylabel('Count')
plt.show()


X4 = X[['MarginalAdhesion']].values

plt.hist(X4)
plt.xlabel('MarginalAdhesion')
plt.ylabel('Count')
plt.show()


X5 = X[['EpithelialCellSize']].values

plt.hist(X5)
plt.xlabel('EpithelialCellSize')
plt.ylabel('Count')
plt.show()


X6 = X[['BareNuclei']].values

plt.hist(X6)
plt.xlabel('BareNuclei')
plt.ylabel('Count')
plt.show()


X7 = X[['BlandChromatin']].values

plt.hist(X7)
plt.xlabel('BlandChromatin')
plt.ylabel('Count')
plt.show()


X8 = X[['NormalNucleoli']].values

plt.hist(X8)
plt.xlabel('NormalNucleoli')
plt.ylabel('Count')
plt.show()


X9 = X[['Mitoses']].values

plt.hist(X9)
plt.xlabel('Mitoses')
plt.ylabel('Count')
plt.show()


Y = X[['Class']].values

plt.hist(Y)
plt.xticks(range(0, 2))
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
