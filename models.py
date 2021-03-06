import tensorflow as tf
from keras.models import Sequential
from keras import layers, Input
from keras.metrics import MeanSquaredError
import numpy as np

from constants import ACTIVATION_DICT
from utils import timing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')


class ModelBase:

    def __init__(self):
        self.prediction = None
        self.history = None
        self.t = np.zeros(16)  # length of input plaintext

        self._model = Sequential()

    def compile(self):
        self._model.compile(loss='mse',
                            optimizer=tf.keras.optimizers.Adam(lr=0.001),
                            metrics=[MeanSquaredError()])
        print(self._model.summary())

    @timing
    def fit(self, x_train, y_train, n_epochs):
        self.history = self._model.fit(x_train, y_train, epochs=n_epochs, verbose=0)
        return self.history

    def predict(self, x_train):
        self.prediction = self._model.predict(x_train)
        # scale and round predicted values
        predicted_integer = self.prediction * 256
        predicted_integer = [round(i) for i in predicted_integer[0]]
        # float and scaled int values
        return self.prediction, predicted_integer

    def reverse(self):
        # diff per model
        pass

    # scale back the result of decryption
    # can be used without an object of a class
    @staticmethod
    def _scale_back(t):
        scaled_back = t[0] * 256
        arr = scaled_back.tolist()
        # round results
        deciphered = [round(i) for i in arr]

        return deciphered

    def get_model(self):
        return self._model


class ModelMain(ModelBase):
    """
    A class for adding hidden layers of a neural network in iterative manner.
    It allows to inverse the NN and to use it for text decryption. \n
    Passed activation functions are applied after each hidden layer.
    If activation function is passed without specifying number of layers,
    n_hidden is set to 1.
    If activation functions are passed in a list,
    n_hidden is determined automatically or must be equal to the length of an array.

    :param n_hidden: number of hidden layers in feed forward neural network (may be 0).
    :param act_dict: dictionary of structure {'name': (activation function, inverse activation function)}.
    :param act_func: name of the activation function from act_dict dictionary, can be a string or a list of strings.
    :param lr: learning rate.
    """

    def __init__(self, n_hidden: int = None,
                 act_dict: dict = ACTIVATION_DICT,
                 act_func: str or list[str] = 'linear',
                 lr: float = 0.001) -> None:

        super().__init__()
        self.n_hidden = n_hidden
        self.lr = lr
        self.act_dict = act_dict
        self.act_func = act_func

        # if n_hidden parameter was not passed
        # set n_hidden = 1 for string
        # or set n_hidden = length of a list
        if self.n_hidden is None and isinstance(self.act_func, (list, np.array)):
            self.n_hidden = len(act_func)
        elif self.n_hidden is None and isinstance(self.act_func, str):
            self.n_hidden = 1

        # if name of only one activation function is passed
        # then apply it after each hidden layer
        if isinstance(self.act_func, str):
            if self.act_func in self.act_dict:
                self.act_func = [self.act_func] * n_hidden
            else:
                raise AttributeError('Passed unknown activation function!')

        # LAST LAYER - SIGMOID
        # in case of 0 hidden layers we get just 1 weight matrix
        self.act_func.append('sigmoid')
        # print(self.act_func)

        # if names are passed in a list
        # check is functions exist
        # apply after each corresponding layer
        if isinstance(self.act_func, (list, np.ndarray)):
            if self.n_hidden != len(self.act_func) - 1:
                raise AttributeError(f'Wrong length of sequence. '
                                     f'Given {len(self.act_func) - 1} activation function(s) instead of {self.n_hidden}.')

            for func in self.act_func:
                if func in self.act_dict:
                    self._model.add(
                        layers.Dense(16, activation=self.act_dict[func][0])
                    )
                else:
                    raise AttributeError('Passed unknown activation function!')

        else:
            raise AttributeError('act_func must be a str or list[str]!')

    def compile(self):
        self._model.compile(loss='mse',
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                            metrics=[MeanSquaredError()])
        self._model.build(input_shape=(None, 16))
        print(self._model.summary())

    def reverse(self):

        # t - temporary variable
        t = self.prediction

        for i in range(self.n_hidden, -1, -1):
            # reverse list of activation functions
            # select inverse of the functions
            inverse_f = self.act_dict[self.act_func[i]][1]
            t = inverse_f(t)
            # subtract bias
            bias = self._model.layers[i].get_weights()[1]
            t = t - bias
            # multiply by inverse weight matrix
            weights = self._model.layers[i].get_weights()[0]
            t = np.dot(t, np.linalg.inv(weights))

        return self._scale_back(t)
