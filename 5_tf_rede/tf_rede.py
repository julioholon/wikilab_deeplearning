# -*- coding: utf-8 -*-
"""
Rede neural MLP com Tensorflow

Fonte: https://github.com/aymericdamien/TensorFlow-Examples/

@author: Julio L R Monteiro
"""

# Importando as bibliotecas
import tensorflow as tf

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

# parâmetros do tensorflow para entrada e saída
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Variáveis para armazenar os pesos e erros
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construir o modelo
pred = multilayer_perceptron(x, weights, biases)

# Função de custo e otimizador
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Inicializar as variáveis
init = tf.global_variables_initializer()

# Inicializar a sessão
with tf.Session() as sess:
    sess.run(init)

    # Ciclo de treinamento
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Passar por todos os lotes de treinamento
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Executar atimizador e custo
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Computar o custo médio
            avg_cost += c / total_batch

        # Mostrar a execução de cada epoch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

    # Testar o modelo
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculo da precisão
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))