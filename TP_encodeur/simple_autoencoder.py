import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

plt.ion() # matplotlib en mode interactif

# Parametres d'apprentissage ####
num_epochs = 50
batch_size = 128
learning_rate = 1e-3
coeff_bruit = 0.3
#################################

# Classe qui definit un autoencoder simple.
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
############################################

# Chargement du dataset MNIST ################################################
# mettre download=True si il n'est pas deja telecharge
mnist_train = MNIST('./data', transform=transforms.ToTensor(), download=False, train=True)
mnist_test = MNIST('./data', transform=transforms.ToTensor(), download=False, train=False)

# Visualisation d'une imagette du dataset ###################################
img_test = mnist_test.test_data[0,:,:].numpy() # on selectionne l'exemple n°0
plt.imshow(img_test, interpolation='none', cmap=plt.cm.gray)
plt.waitforbuttonpress()

# Exemple d'ajout de bruit sur l'imagette ##################################
bruit = np.random.normal(loc=0.0, scale=1.0, size=img_test.shape)
img_test = img_test.astype(float) / 255. # uint8 vers float entre 0. et 1.
img_bruitee = img_test + coeff_bruit * bruit # ajout du bruit a l'image
img_bruitee = np.clip(img_bruitee, 0., 1.) # force les valeurs entre 0. et 1.
img_bruitee = (img_bruitee * 255).astype(np.uint8) # retour en uint8
plt.imshow(img_bruitee, interpolation='none', cmap=plt.cm.gray)
plt.waitforbuttonpress()

# Ajout d'un bruit gaussien sur les imagettes de test ########################
test_data_numpy = mnist_test.test_data.numpy()
test_data_numpy = test_data_numpy.astype(float) / 255.
bruit = np.random.normal(loc=0.0, scale=1.0, size=test_data_numpy.shape)
test_data_numpy += coeff_bruit * bruit
test_data_numpy = np.clip(test_data_numpy, 0., 1.)
test_data_numpy = (test_data_numpy * 255).astype(np.uint8)
mnist_test.test_data = torch.from_numpy(test_data_numpy)
plt.imshow(mnist_test.test_data[0,:,:], interpolation='none', cmap=plt.cm.gray)
plt.waitforbuttonpress()

# Entrainement du modele sur les images train (non bruitees !) ###########
trainloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

print('Entrainement...')
model = autoencoder().cpu() # charger le modele dans le CPU
criterion = nn.MSELoss() # fonction de cout
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # choix de l'optimisation

for epoch in range(num_epochs):
    print('debut epoch {0}'.format(epoch))
    for data in trainloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cpu()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))

torch.save(model.state_dict(), './sim_autoencoder.pth')

# test dans le modele ########################################################
print('Debruitage...')
testloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
model.eval()
f,ax=plt.subplots(2)
for data in testloader:
    img, _ = data
    img = img.view(img.size(0), -1)
    out = model(img)
    # affichage
    ax[0].imshow(np.reshape(img[-1,:], (28, 28)), cmap='gray', interpolation='none')
    ax[1].imshow(np.reshape(out.data.numpy()[-1,:], (28, 28)), cmap='gray', interpolation='none')
    plt.waitforbuttonpress()
