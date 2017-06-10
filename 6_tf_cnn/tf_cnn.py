# Convolutional Neural Network
# -*- coding: utf-8 -*-
"""
Rede neural Convolucional com Tensorflow

Fonte: Curso SuperDataScience

@author: Julio L R Monteiro
"""

# Importando as bibliotecas do Keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando a CNN
cnn = Sequential()

# Passo 1 - Convolução
cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Agrupamento (Pooling)
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Adicionando outra camada de convolução
cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Achatamento
cnn.add(Flatten())

# Passo 4 - Conexão
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dense(1, activation = 'sigmoid'))

# Compilando a rede
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Pré-processamento de imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Treinando a rede com as imagens
cnn.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)