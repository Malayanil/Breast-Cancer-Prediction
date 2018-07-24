import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('biopsy.csv')

X = df[['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class']]


X1 = X[['ClumpThickness']].values
Y1 = X[['Class']].values

plt.scatter(X1,Y1, label='Dependency')
plt.xlabel('ClumpThickeness')
plt.ylabel('Class')
plt.legend()
plt.show()


X2 = X[['UniformityOfCellSize']].values
Y2 = X[['Class']].values

plt.scatter(X2,Y2, label='Dependency')
plt.xlabel('UniformityOfCellSize')
plt.ylabel('Class')
plt.legend()
plt.show()


X3 = X[['UniformityOfCellShape']].values
Y3 = X[['Class']].values

plt.scatter(X3,Y3, label='Dependency')
plt.xlabel('UniformityOfCellShape')
plt.ylabel('Class')
plt.legend()
plt.show()


X4 = X[['MarginalAdhesion']].values
Y4 = X[['Class']].values

plt.scatter(X4,Y4, label='Dependency')
plt.xlabel('MarginalAdhesion')
plt.ylabel('Class')
plt.legend()
plt.show()


X5 = X[['EpithelialCellSize']].values
Y5 = X[['Class']].values

plt.scatter(X5,Y5, label='Dependency')
plt.xlabel('EpithelialCellSize')
plt.ylabel('Class')
plt.legend()
plt.show()


X6 = X[['BareNuclei']].values
Y6 = X[['Class']].values

plt.scatter(X6,Y6, label='Dependency')
plt.xlabel('BareNuclei')
plt.ylabel('Class')
plt.legend()
plt.show()


X7 = X[['BlandChromatin']].values
Y7 = X[['Class']].values

plt.scatter(X7,Y7, label='Dependency')
plt.xlabel('BlandChromatin')
plt.ylabel('Class')
plt.legend()
plt.show()


X8 = X[['NormalNucleoli']].values
Y8 = X[['Class']].values

plt.scatter(X8,Y8, label='Dependency')
plt.xlabel('NormalNucleoli')
plt.ylabel('Class')
plt.legend()
plt.show()


X9 = X[['Mitoses']].values
Y9 = X[['Class']].values

plt.scatter(X9,Y9, label='Dependency')
plt.xlabel('Mitoses')
plt.ylabel('Class')
plt.legend()
plt.show()
