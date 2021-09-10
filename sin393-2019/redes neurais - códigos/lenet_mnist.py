# ------------------------------------------------------------------------------
# Prof. João Fernando Mari
# joaofmari@gmail.com
# Universidade Federal de Viçosa - campus Rio Paranaíba
# Sistemas de Informação
# SIN 393 - Introdução à Visão Computacional (2019-2)
# ------------------------------------------------------------------------------
# Referencias:
# ------------
# [1] LeNet – Convolutional Neural Network in Python
#     https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/ 
# ------------------------------------------------------------------------------

# Importar os pacotes necessários
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from keras.datasets import mnist, cifar10
from keras.optimizers import SGD
from keras.utils import np_utils

from sklearn import neural_network, model_selection, datasets, metrics
import matplotlib.pyplot as plt

def build_lenet(n_canais, n_linhas, n_cols, n_classes):
    """
    Constrói o modelo da RNC.
    """
    # Inicialização do modelo
    model = Sequential()
    inputShape = (n_linhas, n_cols, n_canais)

    # Verifica se a ordem dos canais está de acordo com o Backend.
    if K.image_data_format() == "channels_first":
        inputShape = (n_canais, n_linhas, n_cols)

    # Define o primeiro conjunto de camadas (CONV => ACTIVATION => POOL)
    #model.add(Conv2D(20, 5, padding="same", input_shape=inputShape))
    model.add(Conv2D(filters=20, kernel_size=5, padding="same", input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Define o segundo conjunto de camadas (CONV => ACTIVATION => POOL)
    model.add(Conv2D(filters=50, kernel_size=5, padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Define a primeira camada de neurônios completamente conectados (FC => ACTIVATION).
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    # Define a segunda camada de neurônios completamente conectados (FC).
    model.add(Dense(units=n_classes))

    # Define o classificador softmax.
    model.add(Activation("softmax"))

    return model


if __name__ == '__main__':    

    print("Baixando o conjunto de dados...")
    # MNIST
    # =====
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    n_lin = 28
    n_col = 28
    n_canais = 1

    # CIFAR
    # =====
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # n_lin = 32
    # n_col = 32
    # n_canais = 3

    # Ajustar a ordem dos canais nas imagens de acordo com o Backend.
    if K.image_data_format() == "channels_first":
        X_train = X_train.reshape((X_train.shape[0], n_canais, n_lin, n_col))
        X_test = X_test.reshape((X_test.shape[0], n_canais, n_lin, n_col))
    else:
        X_train = X_train.reshape((X_train.shape[0], n_lin, n_col, n_canais))
        X_test = X_test.reshape((X_test.shape[0], n_lin, n_col, n_canais))
    
    # Ajusta os valores dos pixels para o intervalo entre 0 e 1.
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Converte a representação dos rótulos para categórico.
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    # Define o otimizador e compila o modelo.
    print("Compilando o modelo...")
    opt = SGD(lr=0.01)
    model = build_lenet(n_canais=n_canais, n_linhas=n_lin, n_cols=n_col, n_classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("Treinando o modelo...")
    model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1)

    # Avaliação do modelo.
    print("Avaliando o modelo...")
    (loss, accuracy) = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
    print("Acuracia: {:.2f}%".format(accuracy * 100))

    # Predição de todos os elementos no conjunto de testes.
    y_pred = model.predict(X_test)

    # Imprime a matriz de confusão e o relatório de classificação.
    print('\nMatriz de confusão:')
    print(metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

    print('\nRelatório de classificação:')
    print(metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
