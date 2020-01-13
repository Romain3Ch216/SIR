# MLP sur données synthétisées

import numpy as np
import matplotlib.pyplot as plt

from neuralnetwork import *

# Chargement de la base d'apprentissage:
BaseApp = np.load('BaseApp_1.npy')
LabelsAppD = np.load('LabelApp_1.npy')
TargetApp = label2target(LabelsAppD)

[NbInput, NbEx] = BaseApp.shape
ClassNbr = np.amax(LabelsAppD) + 1
NbOutput = ClassNbr

# Définition du réseau:
NbHCell = 5
it = 1000
lr = 0.0001
[w1, w2] = mlp2def(NbInput, NbHCell, NbOutput)

# apprentissage:
[nw1, nw2, L] = mlp2train(BaseApp, TargetApp, w1, w2, lr, it )

# affichage:
axe_x=np.linspace(1,it,it)
plt.plot(axe_x,L)
plt.ylabel('Coût quadratique')
plt.xlabel("itérations d'apprentissage")

# score (75,01%):
y = mlp2run( BaseApp, nw1, nw2)
LabelsApp = mlpclass(y)
[score, rate] = score(LabelsApp, LabelsAppD)
print( "Taux de reconnaissance: " + str(rate) + "%" )

plt.show()
