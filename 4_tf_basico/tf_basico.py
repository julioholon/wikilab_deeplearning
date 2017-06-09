# -*- coding: utf-8 -*-
"""
Indrodução do TensorFlow

@author: Julio L R Monteiro
"""

import tensorflow as tf

# Definindo constantes
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

node3 = tf.add(node1, node2)

# Criando uma sessão tradicional
sess = tf.Session()
print(sess.run([node1, node2]))

# Criando uma sessão interativa
intsess = tf.InteractiveSession()
print("soma (node3): ",node3.eval())

# Declarando variáveis
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # "+" significa o mesmo que tf.add(a, b)

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

b0 = tf.Variable([-.3], tf.float32)
b1 = tf.Variable([.3], tf.float32)
x = tf.placeholder(tf.float32)
y_pred = b0 + b1*x

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred, {x:[1,2,3,4]}))

# Cálculo do erro quadrático
y = tf.placeholder(tf.float32)
error = tf.reduce_sum(tf.square(y_pred - y))
print(sess.run(error, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(error)
sess.run(init) # reinicializa as variáveis
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([b0, b1]))