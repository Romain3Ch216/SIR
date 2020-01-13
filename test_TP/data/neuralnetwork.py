
# neuralnetwork: librairie programmation MLP "à la main"

import numpy as np

# fonction sigmoide à soerties dans [-1,+1]
def sigmo(v):
	# pour éviter les NAN:
	np.place(v, v>100, 100)
	np.place(v, v<-100, -100)
	# calcul:
	v2 = np.exp(-2*v)
	return ( 1 - v2 ) / ( 1 + v2 );

# fonction sigmoide dérivée
def sigmop(v):
	x = sigmo(v);
	return 1 - x*x;

# définition d'un réseau 1 couche à n entrées et m sorties
def mlp1def(n, m):
	# tirage aléatoire des poids initiaux 
	result = np.random.rand(m, n+1)*2-1
	return result

# calcul des estimations d'un réseau 1 couche sur les entrées x
def mlp1run(x, w):
	# travail dans l'expace étendu (les seuils sont vus comme des poids w0)
	[NbInput, NbEx] = x.shape 
	x = np.concatenate( ( np.ones((1, NbEx)), x ) )
	# calcul
	return sigmo( w @ x )

# classification par algorithme WTA (classes entre 0 et N-1 où N désigne le nombre de classes)
def mlpclass(y):
	[ClassNbr, ExNbr] = y.shape
	key = np.transpose( np.argsort( np.transpose(y) ) );
	return key[ClassNbr-1]

# calcul du score en nombre d'exemples bien classés sur le total d'exemples
# Labels: labels estimés
# LabelsD: labels désirés
def score(Labels, LabelsD):
	ExNbr = Labels.size
	r = sum( Labels==LabelsD )
	return np.array([r, np.around(r*100/ExNbr, 2)])

# vecteur de labels vers vecteurs cibles pour une base d'exemples
def label2target(Labels):
	ExNbr = Labels.size;
	ClassNbr = np.amax(Labels) + 1
	Target = -np.ones((ClassNbr,ExNbr))
	for i in range(ExNbr):
		Target[Labels[i],i] = +1
	return Target

# calcul du vecteur erreur en sortie d'un réseau
# y: vecteur, matrice des sorties du réseau
# target: vecteur, matrice des sortiers désirées
def mlperror(y, target):
	return y - target

# erreur quadratique
def sqrerror(error):
	return np.mean(error**2) 

# apprentissage d'un réseau perceptron
def mlp1train(x, target, w, lr, it):
	# nombre d'exemples et passage à l'espace étendu:
	[NbInput, NbEx] = x.shape
	x = np.concatenate( ( np.ones((1, NbEx)), x ) )

	# vecteur des erreurs quadratiques:
	L = np.zeros(it)

	# iérations:
	for i in range(it):

		# propagation:
		v = w @ x
		y = sigmo(v)

		# erreur:
		e = mlperror(y, target)
		L[i] = sqrerror(e)
		delta = e*sigmop(v)

		# apprentissage:
		Dw = delta@np.transpose(x)
		w = w - lr*Dw

	return np.array([w, L])

# définition d'un réseau 1 couche cachée à n entrées et m sorties
# c: nombre de cellules sur la couche cachée
def mlp2def(n, c, m):
	w1 = np.random.rand(c, n+1)*2-1
	w2 = np.random.rand(m, c+1)*2-1
	return np.array([w1, w2])

# estimation des sorties d'un réseau à une couche cachée
def mlp2run(x, w1, w2):
	hy = mlp1run(x,w1)
	return mlp1run(hy, w2)

# apprentissage d'un réseau à une couche cachée - algorithme de rétropropagation du gradient
def mlp2train(x, target, w1, w2, lr, it):
	
	# Nombre d'exemples et extension des entrées:
	[NbInput, NbEx] = x.shape
	x = np.concatenate( ( np.ones((1, NbEx)), x ) )

	# vecteur des erreurs quadratiques:
	L = np.zeros(it)

	for i in range(it):

		# matrice des poids vers couches suivante (sans les biais):
		w12 = np.delete(w2,0,axis=1)

		# propagation:
		v1 = w1 @ x
		y1 = sigmo(v1)
		y1 = np.concatenate( ( np.ones((1, NbEx)), y1 ) )

		v2 = w2 @ y1
		y2 = sigmo(v2)

		# erreur couche de sortie:
		e = mlperror(y2, target)
		L[i] = sqrerror(e)

		# rétropropagation de l'erreur
		delta2 = e*sigmop(v2)
		delta1 = ( np.transpose(w12)@delta2 ) * sigmop(v1)

		# apprentissage:
		Dw2 = delta2@np.transpose(y1)
		Dw1 = delta1@np.transpose(x)
		w1 = w1 - lr*Dw1
		w2 = w2 - lr*Dw2

	return np.array([w1, w2, L])




