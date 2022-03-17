from Crypto.Cipher import AES, DES
from Crypto.Hash import SHA256, MD5
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from time import time
import tensorflow_probability as tfp

BLOCK_SIZE = 16
PAD_SYMBOL = ' '

plt.style.use('default')


# ---------------------------------------------  Helper Functions  -----------------------------------------------

# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # f.__name__ returns 'wrap' without @wraps(f)
        print_with_divider(f'func: {f.__name__} took: {(te-ts)} sec')
        return result
    return wrap


# Convert string of bytes to string of bits.
def bytes_to_bits(bytes_arr):
    """
    :param bytes_arr: b'\xcf\xe6\xf7\xda\xab\x9fw\xb7\xc2GM&5Y\x88\x99'
    :return: 110011111110011011110111 ...
    """
    bits_arr = []
    for byte in bytes_arr:
        bits_arr.append(bin(byte)[2:].zfill(8))
    return ''.join(bits_arr)


# Pad or crop string to BLOCK_SIZE.
def pad(string):
    """
    :param string: str of any length
    :return: str of length BLOCK_SIZE
    """
    if len(string) == BLOCK_SIZE:
        return string
    elif len(string) > BLOCK_SIZE:
        # crop
        return string[:BLOCK_SIZE]
    else:
        # pad
        return string + (BLOCK_SIZE - len(string) % BLOCK_SIZE) * PAD_SYMBOL


# Encrypt plaintext message with AES.
def encrypt(msg, password='nopassword'):
    """
    :param msg: string of length that ca be divided by 16
    :param password: string of any length
    :return: encrypted text in bytes, e.g. b'\xcf\xe6...
    """
    # Use md5 to stretch / hash key into 16 byte value.
    # Passed parameter should be in bytes.
    hash_obj = MD5.new(password.encode('cp437'))
    h_key = hash_obj.digest()

    # Encrypt message.
    cipher = AES.new(h_key, AES.MODE_ECB)
    result = cipher.encrypt(msg.encode('cp437'))

    return result


# Decrypt encrypted message with AES.
def decrypt(msg, password='nopassword'):
    """
    :param msg: encrypted text in bytes, e.g. b'\xcf\xe6...
    :param password: string of any length
    :return: decrypted plaintext string
    """
    # Use md5 to stretch / hash key into 16 byte value.
    # Passed parameter should be in bytes.
    hash_obj = MD5.new(password.encode('cp437'))
    h_key = hash_obj.digest()

    # Decrypt message.
    decipher = AES.new(h_key, AES.MODE_ECB)
    plain_text = decipher.decrypt(msg).decode('cp437')

    # Uncomment for deleting padding symbols.
    # Index where padding symbols start.
    # In this case pad symbol should be distinct.
    # pad_index_first = plain_text.find(PAD_SYMBOL)
    # result = plain_text[:pad_index_first]

    return plain_text


# Calculate bit error.
def bit_error(test, predicted):
    """
    :param test: array of integer values of test symbols [112, 213, 54, ...
    :param predicted: array of integer values of predicted symbols [111, 218, 54, ...
    :return: relative bit error, absolute bit error
    """
    predicted = bytes_to_bits(predicted)
    predicted = [int(i) for i in predicted]

    test = bytes_to_bits([b for b in test])
    test = [int(i) for i in test]

    if len(test) != len(predicted):
        raise Exception('Different length of argument arrays!')
    summation = 0
    for i in range(0, len(test)):
        summation += ((test[i] + predicted[i]) % 2)

    return summation / len(test), summation


# Print divider before regular print.
def print_with_divider(*args, **kwargs):
    print('=' * 65)
    print(*args, **kwargs)


def plot_loss_history(history):
    """
    :param history: log from model.fit() --> history.history['loss']
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Convert bytes to integers, scale, put in 2d array.
def prepare_for_neural_network(bytes_text):
    """
    :param bytes_text: text in bytes, e.g. b'\xcf\xe6...
    :return: scaled to [0, 1] np.array
    """
    temp = [b for b in bytes_text]
    prepared = np.array([[i / 256 for i in temp]], dtype='float64')
    return prepared


# ------------------------------------------  Inverse Activation Functions  ------------------------------------------

# Sigmoid
def sigmoid_inv(x):
    """
    sigmoid(y) = 1 / (1 + exp(-x))
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1 - x))


# Softplus
def softplus_inv(x):
    """
    softplus(x) = log(1 + exp(x))
    softplus^(-1)(x) = log(exp(x) - 1.)
    :param x: numpy array
    :return: numpy array
    """
    return tfp.math.softplus_inverse(x).numpy()


# Parametric ReLU
def prelu_inv(x, alpha=0.3):
    """
    PRelu(x) = max(0, alpha * x)
    PRelu^(-1)(x) = x / alpha if x < 0 else x
    :param x: np.array
    :param alpha: float [0, 1]
    :return: np.array
    """
    return np.where(x < 0, x / alpha, x)


# Parametric ReLU
def prelu_inv_01(x, alpha=0.1):
    """
    PReLU(x) = max(0, alpha * x)
    PReLU^(-1)(x) = x / alpha if x < 0 else x
    :param x: np.array
    :param alpha: float [0, 1]
    :return: np.array
    """
    return np.where(x < 0, x / alpha, x)


# Leaky ReLU
def leaky_relu_inv(x, alpha=0.01):
    """
    Leaky ReLU = Parametric ReLU with alpha = 0.01
    :param x: np.array
    :param alpha: float [0, 1]
    :return: np.array
    """
    return np.where(x < 0, x / alpha, x)


# ELU
def elu_inv(x, alpha=0.1):
    """
    ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    ELU^(-1)(x) = x if x > 0 else log(x / alpha + 1)
    :param x: np.array
    :param alpha: float [0, 1]
    :return: np.array
    """
    return np.where(x < 0, np.log(x / alpha + 1), x)
