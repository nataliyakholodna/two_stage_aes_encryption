from utils import sigmoid_inv, softplus_inv, leaky_relu_inv, elu_inv, prelu_inv, prelu_inv_01
from keras import layers
import numpy as np

'''
Dictionary in a format:
{'activation function name': (activation function as argument in keras.layers,
                              inverse activation function)}
'''
ACTIVATION_DICT = {'linear':    ('linear', lambda x: x),
                   'sigmoid':   ('sigmoid', sigmoid_inv),
                   'softplus':  ('softplus', softplus_inv),
                   'tanh':      ('tanh', np.arctanh),
                   'PReLU α=0.1':   (layers.LeakyReLU(alpha=0.1), prelu_inv_01),
                   'PReLU α=0.3':   (layers.LeakyReLU(), prelu_inv),
                   'Leaky ReLU':    (layers.LeakyReLU(alpha=0.01), leaky_relu_inv),
                   'ELU α=0.1':     (layers.ELU(alpha=0.1), elu_inv)
                   }

