# ------------------------------------------------------------------------------
# Prof. João Fernando Mari
# joaofmari@gmail.com
# Universidade Federal de Viçosa - campus Rio Paranaíba
# Sistemas de Informação
# SIN 393 - Introdução à Visão Computacional (2019-2)
# ------------------------------------------------------------------------------
# References:
# [1] Neural network models (supervised)
#     https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# [2] Visualization of MLP weights on MNIST¶
#     https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
# ------------------------------------------------------------------------------

# Importa as bibliotecas necessárias.
import numpy as np
from sklearn import neural_network, model_selection, datasets, metrics
import matplotlib.pyplot as plt

# Baixa o conjunto de dados MNIST.
mnist = datasets.fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

# Ajusta os valores dos pixels para o intervalo entre 0 e 1.
X = X / 255.

# Separa um conjunto para treinamento e um conjunto para testes.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

# Número de amostras de treinamento
n_amostras = X.shape[0]
# Número de classes
n_classes = y.shape

# TEST
print(y.shape)

# Constrói o modelo
clf = neural_network.MLPClassifier(hidden_layer_sizes=(200, 100, 50), 
                                   random_state=42,
                                   verbose=10)
print(clf)

# Treina o modelo.
clf.fit(X_train, y_train)

# Testa o modelo.
y_pred = clf.predict(X_test)

print('\nMatriz de confusão:')
print(metrics.confusion_matrix(y_test, y_pred))

print('\nRelatório de classificação:')
print(metrics.classification_report(y_test, y_pred))

print('Coeficientes:')
print(len(clf.coefs_))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap='gray', vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
