import tensorflow as tf
from keras.models import Sequential
from keras import layers, Input
from keras.metrics import MeanSquaredError
import numpy as np
from utils import timing
import tensorflow_probability as tfp
import os

from utils import sigmoid_inv, softplus_inv, leaky_relu_inv, elu_inv, prelu_inv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.set_floatx('float64')


class ModelBase:

    def __init__(self):
        self.prediction = None
        self.history = None
        self.t = np.zeros(16)  # length of input plaintext

        self.model = Sequential()

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer=tf.keras.optimizers.Adam(lr=0.001),
                           metrics=[MeanSquaredError()])
        print(self.model.summary())

    def fit(self, x_train, y_train, n_epochs):
        self.history = self.model.fit(x_train, y_train, epochs=n_epochs, verbose=0)
        return self.history

    def predict(self, x_train):
        self.prediction = self.model.predict(x_train)
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
        return self.model


# -------------------------------------------------   1   -------------------------------------------------
class Model1(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16))
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=[MeanSquaredError()])

    @timing
    def reverse(self):
        ## reverse layer 4
        t = sigmoid_inv(self.prediction)
        t = t - self.model.layers[4].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[4].get_weights()[0]))

        ## reverse layer 3
        t = t - self.model.layers[3].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = t - self.model.layers[2].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = t - self.model.layers[1].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = t - self.model.layers[0].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   2   -------------------------------------------------
class Model2(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16))
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(16, activation='sigmoid'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 3
        t = sigmoid_inv(self.prediction)
        t = t - self.model.layers[3].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = sigmoid_inv(t)
        t = t - self.model.layers[2].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = t - self.model.layers[1].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = t - self.model.layers[0].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   3   -------------------------------------------------
class Model3(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16))
        self.model.add(layers.Dense(16, activation='sigmoid'))
        self.model.add(layers.Dense(16, activation='sigmoid'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 3
        t = np.log(self.prediction/(1 - self.prediction))
        t = t - self.model.layers[3].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = np.log(t/(1 - t))
        t = t - self.model.layers[2].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = np.log(t/(1 - t))
        t = t - self.model.layers[1].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = t - self.model.layers[0].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   4   -------------------------------------------------
class Model4(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16))
        self.model.add(layers.Dense(16, activation='tanh'))
        self.model.add(layers.Dense(16, activation='tanh'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 3
        t = np.log(self.prediction/(1 - self.prediction))
        t = t - self.model.layers[3].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = np.arctanh(t)
        t = t - self.model.layers[2].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = np.arctanh(t)
        t = t - self.model.layers[1].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = t - self.model.layers[0].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   5   -------------------------------------------------
class Model5(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16, activation='tanh'))
        self.model.add(layers.Dense(16, activation='tanh'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 2
        t = np.log(self.prediction/(1 - self.prediction))
        t = t - self.model.layers[2].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = np.arctanh(t)
        t = t - self.model.layers[1].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = np.arctanh(t)
        t = t - self.model.layers[0].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   6   -------------------------------------------------
class Model6(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16))
        self.model.add(layers.Dense(16, activation='softplus'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 2
        t = sigmoid_inv(self.prediction)
        t = t - self.model.layers[2].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = softplus_inv(t)
       # print(t)
        t = t - self.model.layers[1].get_weights()[1] # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = t - self.model.layers[0].get_weights()[1] # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   7   -------------------------------------------------
class Model7(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16))
        self.model.add(layers.Dense(16, activation='sigmoid'))
        self.model.add(layers.Dense(16, activation='tanh'))
        self.model.add(layers.Dense(16, activation='softplus'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 4
        t = np.log(self.prediction/(1 - self.prediction))
        t = t - self.model.layers[4].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[4].get_weights()[0]))

        ## reverse layer 3
        t = tfp.math.softplus_inverse(t).numpy()
        t = t - self.model.layers[3].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = np.arctanh(t)
        t = t - self.model.layers[2].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = np.log(t/(1 - t))
        t = t - self.model.layers[1].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = t - self.model.layers[0].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   8   -------------------------------------------------
class Model8(ModelBase):
    def __init__(self):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16, activation='tanh'))
        self.model.add(layers.Dense(16, activation='sigmoid'))

    @timing
    def reverse(self):
        ## reverse layer 3
        t = np.log(self.prediction/(1 - self.prediction))
        t = t - self.model.layers[1].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 2
        t = np.arctanh(t)
        t = t - self.model.layers[0].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


# -------------------------------------------------   10   -------------------------------------------------

class Model10(ModelBase):
    def __init__(self, activation_func='linear'):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16, activation=activation_func))
        self.model.add(layers.Dense(16, activation=activation_func))
        self.model.add(layers.Dense(16, activation=activation_func))
        self.model.add(layers.Dense(16, activation=activation_func))
        self.model.add(layers.Dense(16, activation=activation_func))

        self.model.add(layers.Dense(16, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=[MeanSquaredError()])

    def reverse(self, activation_func_inv=lambda x: x):
        ## reverse layer 4
        t = sigmoid_inv(self.prediction)
        t = t - self.model.layers[5].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[5].get_weights()[0]))

        ## reverse layer 3
        t = activation_func_inv(t)
        t = t - self.model.layers[4].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[4].get_weights()[0]))

        ## reverse layer 3
        t = activation_func_inv(t)
        t = t - self.model.layers[3].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = activation_func_inv(t)
        t = t - self.model.layers[2].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = activation_func_inv(t)
        t = t - self.model.layers[1].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = activation_func_inv(t)
        t = t - self.model.layers[0].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)


class Model11(ModelBase):

    def __init__(self, n_hidden=1, activation_func='linear', lr=0.001):
        """
        NOTE: if pass an activation function,
        this func will be applied to _every_ hidden layer
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.lr = lr

        self.model.add(layers.Dense(16, input_dim=16, activation=activation_func))

        for _ in range(1, self.n_hidden):
            self.model.add(layers.Dense(16, activation=activation_func))

        self.model.add(layers.Dense(16, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           metrics=[MeanSquaredError()])
        # print(self.model.summary())

    def reverse(self, activation_func_inv=lambda x: x):
        """
        :type activation_func_inv: function, not string
        """
        # reverse last layer
        t = sigmoid_inv(self.prediction)
        t = t - self.model.layers[self.n_hidden].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[self.n_hidden].get_weights()[0]))

        for i in range(self.n_hidden - 1, -1, -1):
            t = activation_func_inv(t)
            t = t - self.model.layers[i].get_weights()[1]  # remove bias
            t = np.dot(t, np.linalg.inv(self.model.layers[i].get_weights()[0]))

        return self._scale_back(t)


class Model12(ModelBase):
    def __init__(self, activation_func='linear'):
        super().__init__()
        self.model.add(layers.Dense(16, input_dim=16, activation='tanh'))
        self.model.add(layers.Dense(16, activation=layers.LeakyReLU()))
        self.model.add(layers.Dense(16, activation='softplus'))
        self.model.add(layers.Dense(16, activation=layers.LeakyReLU()))
        self.model.add(layers.Dense(16))

        self.model.add(layers.Dense(16, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=[MeanSquaredError()])

    def reverse(self, activation_func_inv=lambda x: x):
        ## reverse layer 4
        t = sigmoid_inv(self.prediction)
        t = t - self.model.layers[5].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[5].get_weights()[0]))

        ## reverse layer 3
        t = t - self.model.layers[4].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[4].get_weights()[0]))

        ## reverse layer 3
        t = prelu_inv(t)
        t = t - self.model.layers[3].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[3].get_weights()[0]))

        ## reverse layer 2
        t = softplus_inv(t)
        t = t - self.model.layers[2].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[2].get_weights()[0]))

        ## reverse layer 1
        t = prelu_inv(t)
        t = t - self.model.layers[1].get_weights()[1]  # remove bias
        t = np.dot(t,  np.linalg.inv(self.model.layers[1].get_weights()[0]))

        ## reverse layer 0
        t = np.arctanh(t)
        t = t - self.model.layers[0].get_weights()[1]  # remove bias
        t = np.dot(t, np.linalg.inv(self.model.layers[0].get_weights()[0]))

        return self._scale_back(t)





