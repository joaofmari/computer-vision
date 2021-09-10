# ------------------------------------------------------------------------------
# Prof. João Fernando Mari
# joaofmari@gmail.com
# Universidade Federal de Viçosa - campus Rio Paranaíba
# Sistemas de Informação
# SIN 393 - Introdução à Visão Computacional (2019-2)
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, preprocessing,model_selection


class Perceptron():
    """
    Classe que implementa um Perceptron camada simples.
    """

    def __init__(self, max_iter=20, learning_rate=0.01):
        """
        """
        # Número máximo de iterações.
        self.max_iter = max_iter
        # Taxa de aprendizado.
        self.learning_rate = learning_rate

        # Define o valor da semente de geração de números aleatórios.
        np.random.seed(1234)
        
        
    def superficie_decisao(self, str_title):
        """
        Plota a superfície de decisão para a época atual
        """
    
        # Define os pontos extremos do segmento da superficie de decisão plotado.
        # p0 = (x0_min, f_x0_min)
        # p1 = (x0_max, f_x0_max)
        # ----------
        x0_min = X.min() - 1.
        x0_max = X.max() + 1.
        # ----------
        f_x0_min = -(self.w[0] / self.w[1]) * x0_min - (self.b / self.w[1])
        f_x0_max = -(self.w[0] / self.w[1]) * x0_max - (self.b / self.w[1])

        colors = ['r', 'g', 'b', 'y', 'c', 'm']

        plt.figure()
        # Plota o conjunto de treinamento.
        for y in np.unique(self.y):
            plt.scatter(self.X[self.y==y][:,0], self.X[self.y==y][:,1], color=colors[y], label=str(y))

        plt.xlabel('x_0')
        plt.ylabel('x_1')
        plt.legend()
        plt.title(str_title)

        # Limita o espaço visualizado.
        plt.xlim(X.min()-.5, X.max()+.5)
        plt.ylim(X.min()-.5, X.max()+.5)

        # Plota a superfície de decisão.
        plt.plot([x0_min, x0_max], [f_x0_min, f_x0_max], color='b')

        # Mostra a figura.
        plt.show()

    def fit(self, X, y, plot=True):
        """
        Treina o modelo.
        """
        # Define o conjunto de treinamento.
        self.X = X
        self.y = y
        
        # Inicialização dos pesos W
        self.w = np.random.rand(X.shape[1]) 
        print('\nPesos: ')
        print(self.w)

        # Inicialização dos bias b
        self.b = np.random.rand() 
        print('\nBias: ')
        print(self.b)

        # Lista contendo os erros em cada época
        erros_epocas = []

        # Plota a superfície de decisão na tela.
        if plot:
            title_str = str('Inicialização')
            self.superficie_decisao(title_str)
        
        # Épocas de treinamento.
        # ----------------------
        for i in range(self.max_iter):
            print('\n====================')
            print('Época %i' % i)
            print('====================')

            erro_epocas = 0
            
            # Iteração
            for j in range(self.X.shape[0]):
                print('\n--------------------')
                print('Iteração %i' % j)
                print('--------------------')

                # v: combinação linear
                v = np.dot(X[j,:], self.w) + self.b
                print('v:')
                print(v)

                # y^: Função de ativação degrau.
                y_out = np.where(v >= 0., 1, 0)
                print('y_out:')
                print(y_out)

                # Erro
                erro = y[j] - y_out
                print('Erro:')
                print(erro)

                # Erro total na época. Utiliza o erro quadrático.
                erro_epocas = erro_epocas + erro**2

                # Atualização dos pesos W
                self.w = self.w +  self.learning_rate * np.dot(erro, X[j,:])
                print('Pesos: ')
                print(self.w)

                # Atualização dos bias, b
                self.b = self.b +  self.learning_rate * erro.sum()
                print('Bias: ')
                print(self.b)

            # Erro da época.
            erro_epocas = erro_epocas / 2.
            print('\nErro da época:')
            print(erro_epocas)

            # Adiciona o erro desta época à lista de erros.
            erros_epocas.append(erro_epocas)
            
            # Plota a superfície de decisão
            if plot:
                title_str = str('Época %d' % i)
                self.superficie_decisao(title_str)

            # Interromper o treinamento de erro da época for menor do que um limiar pré-determinado.
            if np.abs(erro_epocas) <= 0.01:
                break

        # Plota a evolução do erro.
        plt.figure()
        plt.plot(erros_epocas)
        plt.show()
        
    def fit_vec(self, X, y, plot=True):
        """
        Versão em lote do algoritmo de treinamento do perceptron.
        Nessa função, a cada época todos os objetos de conjunto de treinamento são apresentados ao 
        mesmo tempo. O tamanho do lote é o mesmo tamanho do conjunto de treinamento.
        """
        # Define o conjunto de treinamento.
        self.X = X
        self.y = y
        
        # Inicialização dos pesos W
        self.w = np.random.rand(X.shape[1]) 
        print(self.w)

        # Inicialização dos bias b
        self.b = np.random.rand() 
        print(self.b)

        # Lista com os erros de cada época
        erros_epocas = []

        # Plota a superfície de decisão na tela.
        if plot:
            title_str = str('Inicialização')
            self.superficie_decisao(title_str)
        
        # Épocas de treinamento.
        for i in range(self.max_iter):
            print('====================')
            print('Época %i' % i)
            print('====================')

            # v
            v = np.dot(X, self.w) + self.b
            print('\nv:')
            print(v)

            # y^
            y_out = np.where(v >= 0., 1, 0)
            print('\ny_out:')
            print(y_out)

            # erro
            erro = y - y_out
            print('\nErro:')
            print(erro)

            # erro total na época: erro quadratico
            erro_epocas = (erro**2).sum()
            print('\nErro época:')
            print(erro_epocas)
            erros_epocas.append(erro_epocas)

            # Atualização dos pesos W
            self.w = self.w +  self.learning_rate * np.dot(erro, X)
            print('\nPesos: ')
            print(self.w)

            # Atualização dos bias, b
            self.b = self.b +  self.learning_rate * erro.sum()
            print('\nBias: ')
            print(self.b)

            print('\nErro da época:')
            print(erro_epocas)

            # Plota a superfície de decisão na tela.
            if plot:
                title_str = str('Época %d' % i)
                self.superficie_decisao(title_str)

            # Se o erro na época for menor do que 0.001, o treinamento é interrompido.
            if np.abs(erro_epocas) <= 0.001:
                break

        # Plota a evolução dos erros.
        plt.figure()
        plt.plot(erros_epocas)
        plt.show()

if __name__ == '__main__':
    """
    """
    # Constrói um objeto do tipo Perceptron
    perc = Perceptron(max_iter=40, learning_rate=0.05)

    # ============================================
    # A - Conjunto de dados IRIS
    # Obs.: Descomentar para utilizar.
    # ============================================            
    # iris = datasets.load_iris()
    # # Todas as três classes:
    # # ----------------------
    ### X = iris.data[:, :2]
    ### y = iris.target
    # # Seleciona apenas as classes Setosa (0) e Virginica (2).
    # # -------------------------------------------------------
    # X = iris.data[iris.target<2, :2]
    # y = iris.target[iris.target<2]
    # # Divide conjunto de treino e testes
    # X, X_test, y, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
    # # Normaliza os dados.
    # # -------------------
    # X = preprocessing.scale(X)
    # print('Média e desvio padrão:')
    # print(X.mean(axis=0), X.std(axis=0))

    # ============================================
    # Funções binárias
    # Obs.: Descomentar a função que irá utilizar.
    # ============================================
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # --------------------------------------------                  
    # AND
    # --------------------------------------------
    # y = np.array([0, 0, 0, 1])

    # --------------------------------------------
    # OR
    # --------------------------------------------
    y = np.array([0, 1, 1, 1])

    # --------------------------------------------
    # XOR
    # --------------------------------------------
    # y = np.array([0, 1, 1, 0])

    # Treinamento do Perceptron
    # Obs.: Descomentar a versão do algoritmo que irá utilizar.
    # -------------------------
    # Versão simples.
    perc.fit(X, y)
    # Versão em lote
    ### perc.fit_vec(X, y)