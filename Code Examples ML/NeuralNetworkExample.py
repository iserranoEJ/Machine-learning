import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np
import pandas as pd

n = 500 # Numero de registros en nuestros datos
p = 2 # Numero de caracteristicas que tiene cada registro

#Crear dataset
X,Y = make_circles(n_samples = n, factor = 0.5, noise = 0.05)
Y = Y[:, np.newaxis]
plt.scatter(X[Y[:,0] == 0,0], X[Y[:,0] == 0, 1], c='skyblue')
plt.scatter(X[Y[:,0] == 1,0], X[Y[:,0] == 1, 1], c='salmon')
plt.axis("equal")
plt.show()



# Clase de la capa de la red
class neural_layer():

    def __init__(self, n_conn, n_neur, act_f):
        """
        args: n_conn numero de conexiones en capa;
        n_neur cuantas neuronas hay en la capa;
        act_f funcion de activacion
        """

        self.act_f = act_f
        # ? np.random.rand(1,x) da un vector de x numeros random entre 0 y 1 (primer parametro) 
        self.b = np.random.rand(1, n_neur) * 2 - 1 #Parametro de bias -> vector -> tantos como neuronas en la capa
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1 #Matriz de weights

"""
Funciones de activacion -> funcion por la que pasa la suma ponderada que se realiza en la neurona para introducir no-linearidades
Tipos: Sigmoide, RELU, tangencial hiperbolica...
"""
sigm = (lambda x: 1/(1 + np.exp(-x)),  # Funcion anonima (lambda) sigmoide 
        lambda x: x* (1-x)) # Derivada de funcion sigmoide para usar luego en backpropagation

relu = lambda x: np.maximum(0,x)

_x = np.linspace(-5,5,100)
#plt.plot(_x, sigm[0](_x))
#plt.plot(_x, relu(_x))
#plt.show()

"""
Forma fea de hacerlo
layer0 = neural_layer(p, 4, sigm)
layer1 = neural_layer(4,8,sigm)
"""
# Topology = como va a ser construida la red: la primera capa con p neuronas, la siguiente con 4, luego 8...
# La ultima capa tendra que tener un numero de neuronas acorde con el output que queramos obtener

def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]): # enumerate da el indice y el objeto
        nn.append(neural_layer(topology[l], topology[l+1], act_f)) # meter en la lista neural network las capas con x neuronas segun el vector topology que hayamos definido

    return nn

topology = [p,4,8,16,8,4,1] 
neural_net = create_nn(topology, sigm)

"""
1.- Primero se le muestra a la red neuronal un tipo de dato de entrada y otro de salida,
    la red neuronal procesa hacia delante usando sumas poneradas y funciones de activacion
    Al no estar entrenada devolvera numeros inservibles, pero a medida que se va entrenando se corresponderan
    mas con el resultado deseado

2.- Comparar resultados con el vector real usando la funcion de coste.
    Esto dara un error que se usara para backpropagation -> Derivadas parciales que son las que nos permitiran 
    obtener la informacion necesaria para ejecutar el descenso de gradiente

3.- Descenso de gradiente: Optimizar la funcion de coste y asi entrenar a la red

"""
l2_cost = (lambda Ypredicted, Yreal: np.mean((Ypredicted - Yreal)**2), # Funcion de coste -> Error cuadratico medio
            lambda Ypredicted, Yreal: (Ypredicted - Yreal)) 

        


def train(neural_net,X,Y,l2_cost,lr = 0.5, train = True): #lr = learning rate -> factor por el que se multiplica el vector gradiente en el descenso del gradiente que permite determinar en que grado estamos actualizando los datos
    #Forward pass -> pasar vector de entrada por cada una de las capas haciendo primero la suma ponderada 
    # (cada valor de entrada por su weight corresponiente y luego sumarlos todos) y pasar esa suma por la funcion de activacion
    
    out = [(None, X)]
    for l,layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b # Suma ponderada -> Coger X (valor de entrada), multiplicarlo matricialmente (@) por el weight de la primera capa (nn[0]) + el parametro de bias
        a = neural_net[l].act_f[0](z) # Pasar la suma ponderada por la funcion de activacion

        out.append((z,a))

    if train:
        #Backward pass
        deltas = []

        for l in reversed(range(0,len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]
            
            if l == len(neural_net) - 1:
                #Calcular delta ultima capa
                deltas.insert(0,l2_cost[1](a,Y)* neural_net[l].act_f[1](a))

            else:
                #Calcular delta respecto a capa previa
                deltas.insert(0,deltas[0] @ _W.T * neural_net[l+1].W* neural_net[l].act_f[1](a))
               
            _W = neural_net[l].W

        #Gradient descent
        neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis = 0, keepdims = True) * lr
        neural_net[l].b = neural_net[l].W - out[l][1].T @ deltas[0] * lr
    
    return out[-1][1]




