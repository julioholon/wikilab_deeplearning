# -*- coding: utf-8 -*-
"""
Rede neural MLP com Tensorflow

Dica:
    %env KERAS_BACKEND=tensorflow

@author: Julio L R Monteiro
"""

# Importando as bibliotecas
import tensorflow as tf
import numpy as np

# Importando os dados do MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parametros de aprendizado
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Parâmetros da rede
n_hidden_1 = 256 # primeira camada oculta
n_hidden_2 = 256 # segunda camada oculta
n_input = 784 # entradas (para o MNIST, pois a imagem é 28*28)
n_output = 10 # Numero de classes no MNIST (dígitos 0-9)

#Importando bibliotecas do Keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializando um rede sequencial
mlp = Sequential()

# Adicionando a primeira camada oculta, conectada com as entradas
mlp.add(Dense(units = n_hidden_1, kernel_initializer = 'uniform', activation = 'relu', input_dim = n_input))

# Adicionando a segunda camada oculta
mlp.add(Dense(units = n_hidden_2, kernel_initializer = 'uniform', activation = 'relu'))

# Adicionando a camada de saída
mlp.add(Dense(units = n_output, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilando a rede
mlp.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Treinando a rede
x_train = mnist.train.images
y_train = mnist.train.labels

mlp.fit(x_train, y_train, batch_size = batch_size, epochs = training_epochs)        

# Testando os dados de treinamento
x_test = mnist.test.images
y_test = mnist.test.labels
y_pred = mlp.predict(x_test)

# Filtro para porencializar os melhores
y_pred = np.where(y_pred > 0.99, 1., 0.)

# Gerando a matriz de confusão
from sklearn.metrics import confusion_matrix
y_pred = np.dot(y_pred, range(10))
y_real = np.dot(y_test, range(10))
cm = confusion_matrix(y_real, y_pred)