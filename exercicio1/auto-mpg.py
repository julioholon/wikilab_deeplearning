# -*- coding: utf-8 -*-
"""
Exercício 1

MPG para carros

@author: Julio L R Monteiro
"""

# Bibliotecas padrão
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carregando os dados
dataset = pd.read_csv('auto-mpg.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

## Dados faltando
##
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:])
X[:] = imputer.transform(X[:])

##
## Dados de treinamento e teste
##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ajustando o modelo de regressão aos dados de treinamento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Testando o modelo aprendido com os dados de teste 
y_pred = regressor.predict(X_test)

# Score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) 