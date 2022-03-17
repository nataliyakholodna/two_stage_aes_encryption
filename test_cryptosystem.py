import random
import string

import numpy as np
import tensorflow as tf
from utils import (pad, encrypt,
                   decrypt, bit_error, print_with_divider,
                   plot_loss_history, prepare_for_neural_network)

from models import Model11
from sklearn.metrics import mean_squared_error
from keras_visualizer import visualizer

# Disable warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

tf.keras.backend.set_floatx('float64')

# -----------------------------------------------   Text & Password   -----------------------------------------------

text = pad('add some random text')

# Generate random password
symbols_for_password = string.ascii_uppercase + string.ascii_lowercase + string.digits
password = ''.join(random.choices(symbols_for_password, k=10))

print_with_divider('Password:', password)

# -----------------------------------------------   Encrypt with AES   -----------------------------------------------

cipher_text = encrypt(text, password)

# -----------------------------------------------   Prepare for NN   -----------------------------------------------

# Convert text to an array of floats [0, 1]
X_train = prepare_for_neural_network(text.encode('utf-8'))
print_with_divider('X_train:')
print(X_train)

y_train = prepare_for_neural_network(cipher_text)
print('y_train:')
print(y_train)

# ----------------------------------------------   Encrypt & Decrypt   ----------------------------------------------

model = Model11()
model.compile()

# Train the neural network
history = model.fit(X_train, y_train, n_epochs=50)
plot_loss_history(history)

temp, y_pred = model.predict(X_train)
print_with_divider('Predicted:\n', y_pred, '\nTrue:\n', [b for b in cipher_text])

# Convert encrypted text to string
pred_str_cipher = ''.join([chr(i) for i in y_pred])
test_str_cipher = ''.join([chr(i) for i in [b for b in cipher_text]])

print_with_divider(f'⟶ Encrypted with AES plaintext:\n{test_str_cipher}\n' +
                   f'⟶ Encrypted with neural network plaintext:\n{pred_str_cipher}')

# Decrypt ciphertext
reversed_int = model.reverse()

# ----------------------------------------------   Compare   ----------------------------------------------

print_with_divider('⟶ Plaintext (int):')
print(np.array([b for b in text.encode('cp437')]))

print('⟶ Deciphered text (int)')
print(np.array(reversed_int))

print_with_divider('Text, encrypted with NN and later decrypted with AES:')
print(decrypt(bytes(y_pred), password))

pred_str = ''.join([chr(i) for i in reversed_int])
test_str = ''.join([chr(i) for i in [b for b in text.encode('cp437')]])

print_with_divider(f'⟶ Initial plaintext:\n{test_str}\n⟶ Deciphered plaintext:\n{pred_str}')

if test_str == pred_str:
    print_with_divider('Text was deciphered correctly!')
else:
    print_with_divider('Deciphering error!')

# ----------------------------------------------   Error metrics   ----------------------------------------------

# Calculate relative and absolute bit error
relative, absolute = bit_error([b for b in cipher_text], y_pred)
print_with_divider(f'⟶ Relative error: {relative}\n⟶ Absolute: {absolute} / 128')

# Calculate Mean Squared Error
mse = mean_squared_error(y_train, temp)
print_with_divider(f'MSE: {mse}')

# ----------------------------------------------   Visualize model   ----------------------------------------------

# visualizer(model.get_model(), format='png', view=True, filename='images/model_8')
