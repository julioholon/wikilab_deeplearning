# -*- coding: utf-8 -*-
"""
Pré-processamento de dados

@author: Julio L R Monteiro
"""

# Bibliotecas padrão
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Carregando os dados
##
dataset = pd.read_csv('Dados.csv')
X = dataset.iloc[:, :-1].values # 0:3
y = dataset.iloc[:, 3].values

## Dados faltando
##
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
#X_fit = imputer.transform(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

## Dados numéricos: ajustando a escala
##
# Escala entre 0 e 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,1))
X3 = scaler.fit_transform(X[:, 1:3])

# Escala padrão é -1 a 1
from sklearn.preprocessing import scale
X2 = scale(X[:, 1:3])

## Dados categorizados: codificação One-hot 
##
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])
# Codificação one-hot
onehot_enc = OneHotEncoder(categorical_features = [0])
X = onehot_enc.fit_transform(X).toarray()
# Codificando y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

## Dados de treinamento e teste
##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.2, random_state = 0)
