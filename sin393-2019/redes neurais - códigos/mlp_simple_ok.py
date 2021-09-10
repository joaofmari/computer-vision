# -----------------------------------------------------------------------------
# Prof. João Fernando Mari
# joaofmari@gmail.com
# Universidade Federal de Viçosa - campus Rio Paranaíba
# Sistemas de Informação
# SIN 393 - Introdução à Visão Computacional (2019-2)
# -----------------------------------------------------------------------------
# Implementação didática do algoritmo backpropagation em Python 3 utilizando apenas a biblioteca 
# NumPy. O objetivo deste programa é ilustrar uma única iteração do algoritmo backpropagation, 
# incluindo a etapa de propagação adiante (forward) e de retro-propagação (backward).
# -----------------------------------------------------------------------------
# Referencias:
# ------------
# [1]  Back-Propagation is very simple. Who made it Complicated ?
# https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c 
# -----------------------------------------------------------------------------
 
import numpy as np
import matplotlib.pyplot as plt

class MLP():
    """
    """

    def __init__(self, hidden_layer_sizes=(5, 2), max_iter=20, learning_rate=0.01):
        """
        """
        # Número de neurônios nas camadas escondidas.
        self.hidden_layer_sizes = hidden_layer_sizes
        # Número máximo de iterações.
        self.max_iter = max_iter
        # Taxa de aprendizado.
        self.learning_rate = learning_rate

        # Define o valor da semente de geração de números aleatórios.
        # Permite a reprodutibilidade dos experimentos.
        np.random.seed(1234)

    # ========================================
    # Funções de ativação.
    # ========================================
    def sigmoid(self, v):
        """
        Função de ativação sigmóide.
        """
        y_ = 1. / (1 + np.exp(-v))
        return y_

    def sigmoid_deriv(self, v):
        """
        Derivada da função sigmóide.
        """
        y_d = 1./(1 + np.exp(-v)) * (1 - 1/(1 + np.exp(-v)))
        return y_d

    def relu(self, v):
        """
        Função de ativação ReLu.
        """
        y_ = np.zeros(len(v))
        y_[v > 0] = v[v > 0]
        return y_

    def relu_deriv(self, v):
        """
        Derivada da função ReLu.
        """
        y_d = np.zeros(len(v))
        y_d[v > 0] = 1
        return y_d

    def softmax(self, v):
        """
        Função de ativação Softmax.
        """
        y_ = np.exp(v) / np.exp(v).sum()
        return y_

    def softmax_deriv(self, v):
        """
        Derivada da função de ativação Softmax. 
        """
        v_exp = np.exp(v)
        v_exp_sum = v_exp.sum()
        v_exp_ = v_exp_sum - v_exp

        y_d = (v_exp * v_exp_) / v_exp_sum**2.

        return y_d

    def plot_functions(self, v):
        """
        Plota as funções de ativação e suas derivadas.
        Apenas ReLu e Sigmoide.
        """
        plt.figure()

        plt.subplot(2, 2, 1)
        plt.plot(v, self.sigmoid(v))
        
        plt.subplot(2, 2, 2)
        plt.plot(v, self.sigmoid_deriv(v))

        plt.subplot(2, 2, 3)
        plt.plot(v, self.relu(v))
        
        plt.subplot(2, 2, 4)
        plt.plot(v, self.relu_deriv(v))

        plt.show()

    def erro_medio_quadratico(self, y_, y):
        """
        Não implementado ainda.
        """

    def erro_medio_quadratico_deriv(self, y_, y):
        """
        Não implementado ainda.
        """

    def erro_entropia_cruzada(self, y_, y):
        """
        Função de erro, ou perda: entropia cruzada
        y_ : saída da rede
        y : saida desejada
        """
        cross = -1 * ( (y * np.log10(y_)) + ((1. - y) * np.log10(1. - y_)) ) 
        return cross

    def erro_entropia_cruzada_deriv(self, y_, y):
        """
        Derivada da função entropia cruzada.
        """
        cross_d = -1 * ((y * (1 / y_)) + (1 - y) * (1. / (1. - y_)))
        return cross_d
       
    def fit(self, X, y, W=None, b=None):
        """
        X : numpy.array
        Padrões de entrada. Apenas o primeiro será utilizado no treinamento.

        y : numpy.array
        Rótulos dos padrões de entrada, codificados como one-hot.

        W : lista contendo os pesos em arranjos numpy
        Se W não for fornecido, os pesos serão gerados com valores aleatórios de acordo com os 
        valores em self.hidden_layer_sizes

        b : lista contendo os bias em arranjos numpy
        Se b não for fornecido, os valores dos bias serão gerados aleatóriamente de acordo com os 
        valores em self.hidden_layer_sizes
        """
        # Dimensionalidade dos dados.
        n_dim = X.shape[1]
        # Número de amostras no conjunto de treinamento.
        n_amostras = X.shape[0]
        # Número de classes.
        n_classes = y.shape[1]

        # ================================================================================
        # Inicialização dos pesos (W) e bias (b) com valores aleatórios.
        # Inicialização dos resultados dos somadores (v) e das funções de ativação (y_).
        # ================================================================================
        # Camada 0
        if W is None:
            W_0 = np.random.rand(n_dim, self.hidden_layer_sizes[0])
            b_0 = np.random.rand(self.hidden_layer_sizes[0])
        else:
            W_0 = W[0]
            b_0 = b[0]
        v_0 = np.zeros(self.hidden_layer_sizes[0])
        y_0 = np.zeros(self.hidden_layer_sizes[0])

        # Camada 1
        if W is None:
            W_1 = np.random.rand(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
            b_1 = np.random.rand(self.hidden_layer_sizes[1])
        else:
            W_1 = W[1]
            b_1 = b[1]
        v_1 = np.zeros(self.hidden_layer_sizes[1])
        y_1 = np.zeros(self.hidden_layer_sizes[1])
    
        # Camada 2 (camada de saída)
        if W is None:
            W_2 = np.random.rand(self.hidden_layer_sizes[1], n_classes)
            b_2 = np.random.rand(n_classes)
        else:
            W_2 = W[2]
            b_2 = b[2]
        v_2 = np.zeros(n_classes)
        y_2 = np.zeros(n_classes)

        # Print
        print('\n=======================================')
        print('Inicialização dos pesos')
        print('=======================================')
        print('\nW^0')
        print(W_0)

        print('\nb^0')
        print(b_0)

        print('\nW^1')
        print(W_1)
        print('\nb^1')
        print(b_1)

        print('\nW^2')
        print(W_2)
        print('\nb^2')
        print(b_2)

        #### Épocas: Uma única época
        # Em um exemplo completo, aqui estaria o laço de repetição que controla as épocas de 
        # treinamento.

        #### Iterações: Uma única iteração
        # Em um exemplo completo, aqui estário o laço de repetição que controla as iterações da 
        # época atual.

        # Padrão de entrada.
        x = X[0, :]
        print('\nx:')
        print(x)

        # ================================================================================
        # Forward:
        # ================================================================================
        print('\n========================================')
        print('Forward...')
        print('========================================')
        
        # Camada 0
        # --------
        print('\n---------------------------------------')
        print('Camada 0:')
        print('---------------------------------------')
        # Produto interno
        v_0 = np.dot(x, W_0) + b_0
        # Print
        print('\nv^0:')
        print(v_0)

        # Função de ativação: ReLu
        y_0 = self.relu(v_0)
        print('\ny^0:')
        print(y_0)

        # Camada 1 
        # --------
        print('\n---------------------------------------')
        print('Camada 1:')
        print('---------------------------------------')
        # Produto interno
        v_1 = np.dot(y_0, W_1) + b_1
        print('\nv^1:')
        print(v_1)

        # Função de ativação: sigmoid
        y_1 = self.sigmoid(v_1)
        print('\ny^1:')
        print(y_1)

        # Camada 2 (camada de saída)
        # --------------------------
        print('\n---------------------------------------')
        print('Camada 2 (camada de saída):')
        print('---------------------------------------')
        # Produto interno
        v_2 = np.dot(y_1, W_2) + b_2
        print('\nv^2:')
        print(v_2)

        # Função de ativação: softmax
        y_2 = self.softmax(v_2)
        print('\ny^2:')
        print(y_2)
                
        # Calculo do erro
        # ----------------------------------------
        erro = self.erro_entropia_cruzada(y_2, y)
        print('\nErro: ')
        print(erro)
        
        # ================================================================================
        # Backward
        # ================================================================================
        print('\n========================================')
        print('Backward...')
        print('========================================')

        # ////////////////////////////////////////////////////////////////////////////////
        # Camada 2 (camada de saída)
        # ////////////////////////////////////////////////////////////////////////////////
        print('\n---------------------------------------')
        print('Camada 2 (camada de saída):')
        print('---------------------------------------')
        # --------------------------------------------------------------------------------
        # ∂E/∂W^2 = ∂E/∂y^2 * ∂y^2/∂v^2 * ∂v^2/∂W^2
        # --------------------------------------------------------------------------------
        print('\n∂E/∂W^2 = ∂E/∂y^2 * ∂y^2/∂v^2 * ∂v^2/∂W^2')
        print('-----------------------------------------')

        # ∂E/∂y^2
        dE_dy_2 = self.erro_entropia_cruzada_deriv(y_2, y)
        print('\n∂E/∂y^2:')
        print(dE_dy_2)

        # ∂y^2/∂v^2
        dy_2_dv_2 = self.softmax_deriv(v_2)
        print('\n∂y^2/∂v^2:')
        print(dy_2_dv_2)

        # ∂v^2/∂W^2
        dv_2_dW_2 = np.ones(W_2.shape) * y_1[..., np.newaxis]
        print('\n∂v^2/∂W^2:')
        print(dv_2_dW_2)

        # ∂E/∂W^2:
        # -------- 
        dE_dW_2 = dv_2_dW_2 * dy_2_dv_2 * dE_dy_2
        print('\n∂E/∂W^2:')
        print('--------')
        print(dE_dW_2)

        # Atualização dos pesos da camada 2 (W^2):
        # ---------------------------------------
        W_2 = W_2 - self.learning_rate * dE_dW_2
        print('\nW\'^2 = W^2(t - 1) - η * ∂E/∂W^2')
        print(W_2)

        # ////////////////////////////////////////////////////////////////////////////////
        # Camada 1 (camada escondida):
        # ////////////////////////////////////////////////////////////////////////////////
        print('\n---------------------------------------')
        print('Camada 1 (camada escondida):')
        print('---------------------------------------')
        # --------------------------------------------------------------------------------
        # ∂E^total/∂W^1 = ∂E^total/∂y^1 * ∂y^1\∂v^1 * ∂y^1/∂W^1
        # --------------------------------------------------------------------------------
        print('\n∂E^total/∂W^1 = ∂E^total/∂y^1 * ∂y^1\∂v^1 * ∂y^1/∂W^1')
        print('-----------------------------------------------------')

        # ∂y^1/∂v^1
        dy_1_dv_1 = self.sigmoid_deriv(v_1)
        print('\n∂y^1/∂v^1')
        print(dy_1_dv_1)

        # ∂v^1/∂W^1
        dv_1_dW_1 = np.ones(W_1.shape) * y_0[..., np.newaxis]
        print('\n∂v^1/∂W^1')
        print(dv_1_dW_1)

        # ∂E^total/∂y^1 = ∂E/∂y^2 * ∂y^2/∂v^2 * ∂v^2/∂y^1
        # -----------------------------------------------
        # ∂E/∂y^2 : Já calculado
        # ∂y^2/∂v^2 : Já calculado
        # ∂v_2/∂y_1 : Calcular!!!
        # -----------------------------------------------
        print('\n∂E^total/∂y^1 = ∂E/∂y^2 * ∂y^2/∂v^2 * ∂v^2/∂y^1')
        print('- - - - - - - - - - - - - - - - - - - - - - - -')

        # Print
        print('\ndE_dy^2: (já calculado anteriormente)')
        print(dE_dy_2)

        # Print
        print('\ndy^2_dv^2: (já calculado anteriormente)')
        print(dy_2_dv_2)

        # ∂v^2/∂y^1
        dv_2_dy_1 = W_2
        print('\n∂v^2/∂y^1:')
        print(dv_2_dy_1)
        
        # ∂E^total/∂y^1
        dE_total_dy_1 = dE_dy_2 * dy_2_dv_2 * dv_2_dy_1
        print(dE_total_dy_1)
        dE_total_dy_1 = dE_total_dy_1.sum(axis=1)
        print('\n∂E^total/∂y^1:')
        print('- - - - - - - -')
        print(dE_total_dy_1)

        # ∂E^total/∂W^1 = ∂E^total/∂y^1 * ∂y^1\∂v^1 * ∂y^1/∂W^1
        # -----------------------------------------------------
        dE_total_dW_1 = dE_total_dy_1 * dy_1_dv_1 * dv_1_dW_1
        print('\n∂E^total/∂W^1:')
        print('--------------')
        print(dE_total_dW_1)

        # Atualização dos pesos da camada 1 (W^1):
        # ----------------------------------------
        W_1 = W_1 - self.learning_rate * dE_total_dW_1
        print('\nW\'^1 = W^1 - η * ∂E^total/∂W^1')
        print(W_1)

        # ////////////////////////////////////////////////////////////////////////////////
        # Camada 0 (camada escondida):
        # ////////////////////////////////////////////////////////////////////////////////
        print('\n---------------------------------------')
        print('Camada 0 (camada escondida):')
        print('---------------------------------------')
        # --------------------------------------------------------------------------------
        # ∂E^total/∂W^0 = ∂E^total/∂y^0 * ∂y^0/∂v^0 * ∂v^0/∂W^0
        # --------------------------------------------------------------------------------
        print('\n∂E^total/∂W^0 = ∂E^total/∂y^0 * ∂y^0/∂v^0 * ∂v^0/∂W^0')
        print('-----------------------------------------------------')
        
        # ∂y^0/∂v^0
        dy_0_dv0 = self.relu_deriv(v_0)
        print('\n∂y^0/∂v^0:')
        print(dy_0_dv0)

        # ∂v^0/∂W^0
        dv_0_dW_0 = np.ones(W_0.shape) * x[..., np.newaxis]
        print('\n∂v^0/∂W^0:')
        print(dv_0_dW_0)

        # ∂E^total/∂y^0 = ∂E^total/∂y^1 * ∂y^1/∂v^1 * ∂v^1/∂y^0
        # -----------------------------------------------------
        # ∂E^total/∂y^1 : Já calculado
        # ∂y^1/∂v^1 : Já calculado
        # ∂v^1/∂y^0 : Calcular!!!
        # -----------------------------------------------------
        print('\n∂E^total/∂y^0 = ∂E^total/∂y^1 * ∂y^1/∂v^1 * ∂v^1/∂y^0')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')

        # Print
        print('\n∂E^total/∂y_1: (já calculado)')
        print(dE_total_dy_1)

        # Print
        print('\n∂y^1/∂v^1: (já calculado)')
        print(dy_1_dv_1)

        # ∂v_1/∂y_0
        dv_1_dy_0 = W_1
        print('\n∂v^1/∂y^0:')
        print(dv_1_dy_0)
        
        # ∂E_total/∂y_0
        dE_total_dy_0 = dE_total_dy_1 * dy_1_dv_1 * dv_1_dy_0
        print(dE_total_dy_0)
        dE_total_dy_0 = dE_total_dy_0.sum(axis=1)
        print('\n∂E^total/∂y^0:')
        print('- - - - - - - -')
        print(dE_total_dy_0)

        # ∂E^total/∂W^0 = ∂E^total/∂y^0 * ∂y^0\∂v^0 * ∂y^0/∂W^0
        # -----------------------------------------------------
        dE_total_dW_0 = dE_total_dy_0 * dy_0_dv0 * dv_0_dW_0
        print('\n∂E^total/∂W^0:')
        print('--------------')
        print(dE_total_dW_0)

        # Atualização dos pesos da camada 0 (W^0):
        # ---------------------------------------
        W_0 = W_0 - self.learning_rate * dE_total_dW_0
        print('\nW\'^0 = W^0(t - 1) - η * ∂E^total/∂W^0:')
        print(W_0)


if __name__ == '__main__':
    """
    Descomentar o exemplo que pretende executar!
    """

    # ===============
    # Exemplo em [1]:
    # ===============
    # mlp = MLP(hidden_layer_sizes=(2, 3))
    # mlp.plot_functions(v)
    # X = np.array([[0.1, 0.2, 0.7]])
    # y = np.array([[1.0, 0.0, 0.0]])
    # W_0 = np.array([[0.1, 0.2, 0.3],
    #                 [0.3, 0.2, 0.7],
    #                 [0.4, 0.3, 0.9]] )
    # W_1 = np.array([[0.2, 0.3, 0.5],
    #                 [0.3, 0.5, 0.7],
    #                 [0.6, 0.4, 0.8]] )
    # W_2 = np.array([[0.1, 0.4, 0.8],
    #                 [0.3, 0.7, 0.2],
    #                 [0.5, 0.2, 0.9]] )
    # W = [W_0, W_1, W_2]
    # b_0 = [1., 1., 1.]
    # b_1 = [1., 1., 1.]
    # b_2 = [1., 1., 1.]
    # b = [b_0, b_1, b_2]
    #
    # Treina o modelo. 
    # mlp.fit(X, y, W, b)
    # ---------------

    # ==================
    # Exemplo no README.
    # ------------------
    mlp = MLP(hidden_layer_sizes=(2, 3), learning_rate=0.1)

    X = np.array([[0.1, 0.7]])
    y = np.array([[1.0, 0.0]])
    
    # Treina o modelo.
    mlp.fit(X, y)
    # -----------------
