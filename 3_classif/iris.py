import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)

#from sklearn import tree
#with open("iris.dot", 'w') as f:
#    dot_data = tree.export_graphviz(estimator, 
#                         filled=True, rounded=True,  
#                         special_characters=True,
#                         out_file=f)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# Load data
iris = load_iris()

X = iris.data[:, [0,1]]
y = iris.target

# Train
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0).fit(X, y)

from sklearn import tree
with open("iris.dot", 'w') as f:
    dot_data = tree.export_graphviz(clf, 
                         filled=True, rounded=True,  
                         special_characters=True,
                         out_file=f)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
#cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.Paired)
#plt.axis("tight")
#plt.legend()
plt.show()