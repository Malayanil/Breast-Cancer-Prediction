import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r

df = pd.read_csv('biopsy.csv')

X = df[['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class']]
X = X.values

features = ['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'EpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class']

fig, ax = plt.subplots()        
n=r.randint(1,100)
im =ax.imshow(X[n:(n+10)])

ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(features)))

ax.set_xticklabels(features)
ax.set_yticklabels(features)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax.set_title('Heatmap of Feature\'s CoRelations')
plt.gca().invert_xaxis()
fig.tight_layout()
plt.colorbar(im)
plt.show()
