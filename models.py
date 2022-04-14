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
        predicted_integer = self.prediction * 256
        predicted_integer = [round(i) for i in predicted_integer[0]]
        return self.prediction, predicted_integer

    def reverse(self):
        # diff per model
        pass

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

    def __init__(self, n_hidden: int = None,
                 act_dict: dict = ACTIVATION_DICT,
                 act_func: str or list[str] = 'linear',
                 lr: float = 0.001) -> None:
        """
        """
        super().__init__()
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.n_hidden = len(act_func)

        self.lr = lr
        self.act_dict = act_dict
        self.act_func = act_func

        # if name of only one activation function is passed
        if isinstance(self.act_func, str):
            if self.act_func in self.act_dict:
                self.act_func = [self.act_func] * n_hidden
            else:
                raise AttributeError('Passed unknown activation function!')

        # if names are passed in a list
        if isinstance(self.act_func, (list, np.ndarray)):
            if self.n_hidden != len(self.act_func):
                raise AttributeError(f'Wrong length of sequence. '
                                     f'Given {len(self.act_func)} activation function(s) instead of {self.n_hidden}.')

            for func in self.act_func:
                if func in self.act_dict:
                    self._model.add(
                        layers.Dense(16, activation=self.act_dict[func][0])
                    )
                else:
                    raise AttributeError('Passed unknown activation function!')

        else:
            raise AttributeError('act_func must be a str or list[str]!')

        # output layer
        # in case of 0 hidden layers we get just 1 weight matrix
        self._model.add(layers.Dense(16, activation='sigmoid'))

    def compile(self):
        self._model.compile(loss='mse',
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                            metrics=[MeanSquaredError()])
        self._model.build(input_shape=(None, 16))
        print(self._model.summary())

    def reverse(self):

        # reverse last layer
        sigmoid_inverse = self.act_dict['sigmoid'][1]
        t = sigmoid_inverse(self.prediction)
        bias = self._model.layers[self.n_hidden].get_weights()[1]
        t = t - bias  # remove bias
        weights = self._model.layers[self.n_hidden].get_weights()[0]
        t = np.dot(t, np.linalg.inv(weights))

        for i in range(self.n_hidden - 1, -1, -1):
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
