# -*- coding: utf-8 -*-
"""
Regress�o Linear Simples

@author: Julio L R Monteiro
"""

# Bibliotecas padr�o
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carregando os dados
dataset = pd.read_csv('Salarios.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

##
## Dados de treinamento e teste
##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ajustando o modelo de regress�o aos dados de treinamento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Testando o modelo aprendido com os dados de teste 
y_pred = regressor.predict(X_test)

# Visualizando os dados de treinamento e o resultado
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'lightblue')
plt.title('Salario vs Experiência (Dados de treinamento)')
plt.xlabel('Anos de experiência')
plt.ylabel('Salario')
plt.show()

# Visualizando os dados de teste
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'lightblue')
plt.title('Salario vs Experiência (Dados de teste)')
plt.xlabel('Anos de experiência')
plt.ylabel('Salario')
plt.show()