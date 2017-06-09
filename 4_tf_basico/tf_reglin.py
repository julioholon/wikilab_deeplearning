# -*- coding: utf-8 -*-
"""
Regressão Linear com Tensorflow

@author: Julio L R Monteiro
"""

# Importando as bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy.random as rng

# Carregando os dados
dataset = pd.read_csv('Salarios.csv')
X_in = dataset.iloc[:, :-1].values
y_in = dataset.iloc[:, 1].values

#Ajuste de escala
X_in.reshape(-1,1)
y_in.reshape(-1,1)

##
## Dados de treinamento e teste
##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size = 0.2, random_state = 0)

# Parâmetros do Tensorflow
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Valores aleatórios
b0 = tf.Variable(rng.randn())
b1 = tf.Variable(rng.randn())

# Modelo linear
y_pred = b0 + b1*X

# Função de custo
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2))/(2*X_train.shape[0])

# Gradiente descendente
# As variáveis b0 e b1 são alteráveis pois são trainable=True
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Inicialização de variáveis
init = tf.global_variables_initializer()

# Inicializa a sessão
with tf.Session() as sess:
    sess.run(init)

    # Treinar a massa de treinamento
    for epoch in range(1000): # Número de Epochs
        for (x,y) in zip(X_train, y_train):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Mostra o resultado da otimização
        if (epoch+1) % 100 == 0:
            c = sess.run(cost, feed_dict={X: X_train, Y:y_train})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "b0=", sess.run(b0), "b1=", sess.run(b1))

    # Visualizando os dados de treinamento e o resultado
    plt.scatter(X_train, y_train, color = 'blue')
    plt.plot(X_train, sess.run(b0) + sess.run(b1) * X_train, color = 'lightblue')
    plt.title('Salario vs Experiência (Dados de treinamento)')
    plt.xlabel('Anos de experiência')
    plt.ylabel('Salario')
    plt.show()
    
    # Visualizando os dados de teste
    plt.scatter(X_test, y_test, color = 'blue')
    plt.plot(X_train, sess.run(b0) + sess.run(b1) * X_train, color = 'lightblue')
    plt.title('Salario vs Experiência (Dados de teste)')
    plt.xlabel('Anos de experiência')
    plt.ylabel('Salario')
    plt.show()


#    training_cost = sess.run(cost, feed_dict={X: X_train, Y: y_train})
#    print("Training cost=", training_cost, "b0=", sess.run(b0), "b1=", sess.run(b1), '\n')
#
#    testing_cost = sess.run(
#        tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2*X_test.shape[0]),
#        feed_dict={X: X_test, Y: y_test})  # same function as cost above
#    print("Testing cost=", testing_cost)
#    print("Absolute mean square loss difference:", abs(
#        training_cost - testing_cost))
#
