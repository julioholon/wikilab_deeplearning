# -*- coding: utf-8 -*-
"""
Classificação usando Árvores de Decisão

@author: Julio L R Monteiro
"""

# Bibliotecas padrão
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carregando os dados
dataset = pd.read_csv('Classif.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values


## Dados numéricos: ajustando a escala
##
# Escala entre 0 e 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,1))
X[:, 0:2] = scaler.fit_transform(X[:, 0:2])

## Dados de treinamento e teste
##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Treinando o classificador da árvore de decisão
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Previsão usando os dados de teste
y_pred = classifier.predict(X_test)

## Código para gerar a árvore de decisão num formato que pode ser impresso
##
#from sklearn import tree
#with open("tree.dot", 'w') as f:
#    dot_data = tree.export_graphviz(classifier, 
#                         filled=True, rounded=True,  
#                         special_characters=True,
#                         out_file=f)

# Gerando a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizando os dados de treinamento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
XX, YY = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.1, stop = X_set[:, 0].max() + 0.1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.1, stop = X_set[:, 1].max() + 0.1, step = 0.01))
Z = classifier.predict(np.array([XX.ravel(), YY.ravel()]).T)
Z = Z.reshape(XX.shape)
plt.contourf(XX, YY, Z, alpha=0.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(XX.min(), XX.max())
plt.ylim(YY.min(), YY.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Classificação com Árvore de Decisão (Treinamento)')
plt.xlabel('Frequencia')
plt.ylabel('ENEM')
plt.show()

# Visualizando os dados de teste
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
XX, YY = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.1, stop = X_set[:, 0].max() + 0.1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.1, stop = X_set[:, 1].max() + 0.1, step = 0.01))
Z = classifier.predict(np.array([XX.ravel(), YY.ravel()]).T)
Z = Z.reshape(XX.shape)
plt.contourf(XX, YY, Z, alpha=0.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(XX.min(), XX.max())
plt.ylim(YY.min(), YY.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Classificação com Árvore de Decisão (Teste)')
plt.xlabel('Frequencia')
plt.ylabel('ENEM')
plt.show()