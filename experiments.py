import os
import random
import string
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from models import Model11, Model12

from scipy.stats import sem  # std / √n
from sklearn.metrics import mean_squared_error

from utils import pad, encrypt, bit_error, print_with_divider, prepare_for_neural_network
from utils import sigmoid_inv, softplus_inv, leaky_relu_inv, elu_inv, prelu_inv_01, prelu_inv

from models import Model10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

# ---------------------------------------------   Text & Password   -----------------------------------------------

text = pad('add some random text')

# Generate random password
symbols_for_password = string.ascii_uppercase + string.ascii_lowercase + string.digits
password = ''.join(random.choices(symbols_for_password, k=10))

print_with_divider('Password:', password)

# -------------------------------------------   Encrypt with AES   ---------------------------------------------

cipher_text = encrypt(text, password)

# -------------------------------------------   Prepare for NN   ---------------------------------------------

#  text.encode('utf-8'),  cipher_text
X_train = prepare_for_neural_network(text.encode('utf-8'))
print_with_divider('X_train:')
print(X_train)

y_train = prepare_for_neural_network(cipher_text)
print('y_train:')
print(y_train)

# -------------------------------------------   Activation Functions   ---------------------------------------------

ACTIVATION = ['linear', 'softplus', 'tanh', 'sigmoid', layers.LeakyReLU(alpha=0.1),
              layers.LeakyReLU(), layers.LeakyReLU(alpha=0.01), layers.ELU(alpha=0.1)]

NAMES = ['linear', 'softplus', 'tanh', 'sigmoid', 'PReLU α=0.1', 'PReLU α=0.3', 'Leaky ReLU', 'ELU α=0.1']

INVERSES = [lambda x: x, softplus_inv, np.arctanh, sigmoid_inv,
            prelu_inv_01, prelu_inv, leaky_relu_inv, elu_inv]


#%%
##################################################
''' Mean error / loss, time for n=100 attempts '''
##################################################

# ---------------------------------------------   Result DataFrame   -----------------------------------------------

df = pd.DataFrame(columns=['Function', 'MSE', 'MSE SD', 'Hamming Distance', 'Hamming SD', 'Time Fit'],
                  index=range(len(ACTIVATION)))

# --------------------------------------------------   Main Loop   --------------------------------------------------

for i, _ in enumerate(ACTIVATION):

    mse_arr = []
    rel_err_arr = []
    abs_err_arr = []
    time_fit = []

    for j in tqdm(range(10)):  # hard-coded number of experiments
        model = Model10(ACTIVATION[i])
        model.compile()

        start = datetime.now()
        history = model.fit(X_train, y_train, n_epochs=10)
        stop = datetime.now()

        temp, y_pred = model.predict(X_train)
        reversed_int = model.reverse(INVERSES[i])

        relative, absolute = bit_error([b for b in cipher_text], y_pred)
        mse = mean_squared_error(y_train, temp)

        mse_arr.append(mse)
        rel_err_arr.append(relative)
        abs_err_arr.append(absolute)
        time_fit.append(stop - start)

    print_with_divider(NAMES[i])
    df.loc[i, 'Function'] = NAMES[i]

    # --------------------------------------------------   Mean   -----------------------------------------------------

    # Mean errors per given number of experiments
    mean_mse = np.round(np.mean(mse_arr), 5)
    se_mse = np.round(sem(mse_arr, ddof=0), 5)
    std_mse = np.round(np.std(mse_arr), 5)

    print(f'\nMSE: {mean_mse} +- {se_mse} (SE)')
    print(f'\nSD: {std_mse}')

    df.loc[i, 'MSE'] = f'{mean_mse} +- {se_mse} (SE)'
    df.loc[i, 'MSE SD'] = std_mse

    # ----------------------------------------------   Absolute Error  -------------------------------------------------

    mean_abs = np.round(np.mean(abs_err_arr), 5)
    se_abs = np.round(sem(abs_err_arr, ddof=0), 5)
    std_abs = np.round(np.std(abs_err_arr), 5)

    print(f'\nAbsolute: {mean_abs} +- {se_abs} (SE)')
    print(f'\nSD: {std_abs}')

    df.loc[i, 'Hamming Distance'] = f'{mean_abs} +- {se_abs} (SE)'
    df.loc[i, 'Hamming SD'] = std_abs

    # ----------------------------------------------   Fit Time  --------------------------------------------------

    time_fit_mean = pd.to_timedelta(pd.Series(time_fit)).mean()

    print(f'\nMean time: {time_fit_mean}\n')

    df.loc[i, 'Time Fit'] = time_fit_mean

    print_with_divider()

# df.to_csv('hidden_5_100.csv')

# %%
########################################
''' Plot mean loss for n=10 attempts '''
########################################

plt.figure(figsize=(8, 4))

for i, _ in enumerate(ACTIVATION):

    losses = []

    for j in tqdm(range(10)):
        model = Model11(activation_func=ACTIVATION[i], n_hidden=5)
        model.compile()

        history = model.fit(X_train, y_train, n_epochs=200)
        losses.append(history.history['loss'])

        # ensure NN still can correctly decrypt
        temp, y_pred = model.predict(X_train)
        reversed_int = model.reverse(INVERSES[i])

    # Mean loss per given number of experiments per epoch
    mean_loss = np.mean(losses, axis=0)
    plt.plot(mean_loss, label=NAMES[i])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

# %%
##############################################################
''' Average time to train based on number of hidden layers '''
##############################################################

times = []
MAX_HIDDEN = 50

# Iteratively add hidden layers
for n_hidden in range(1, MAX_HIDDEN):

    avg_time = []

    for i in range(10):

        model = Model11(n_hidden=n_hidden)
        model.compile()

        start = datetime.now()
        history = model.fit(X_train, y_train, n_epochs=200)
        stop = datetime.now()

        temp, y_pred = model.predict(X_train)
        reversed_int = model.reverse()

        # Average training time per given number of hidden layers
        avg_time.append(np.float64(str((stop - start).seconds) +
                        '.' + str((stop - start).microseconds)))

    # Mean time per n_hidden layers
    times.append(np.mean(avg_time))

#%%
# Plot the results
import seaborn as sns

sns.regplot(list(range(1, MAX_HIDDEN)), times)
plt.xlabel('Number of hidden layers')
plt.ylabel('Avg time of training, sec')
plt.show()

#%%
#####################
''' Learning Rate '''
#####################

LR = [0.01, 0.001, 1e-4, 1e-5]

for lr in tqdm(LR):
    model = Model11(n_hidden=3, lr=lr)
    model.compile()

    history = model.fit(X_train, y_train, n_epochs=500)

    # Plot loss per learning rate
    plt.plot(history.history['loss'], label=lr)

    temp, y_pred = model.predict(X_train)
    reversed_int = model.reverse()

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('NN with 3 hidden layers and linear activation function')
plt.legend(title='Learning rate')
plt.show()


#%%
###############################################################
''' Average error / loss based on number of training epochs '''
###############################################################

# -----------------------------------------------   Collect Errors   -----------------------------------------------

epochs = [200]  # 10, 50, 100,

# Losses per number of epochs
MSE_ARR, REL_ERR_ARR, ABS_ERR_ARR = {}, {}, {}

start = datetime.now()

for epoch in epochs:

    print(f'Num epochs: {epoch}')

    mse_arr = []
    rel_err_arr = []
    abs_err_arr = []

    for i in tqdm(range(200)):

        model = Model11(n_hidden=3)
        model.compile()

        history = model.fit(X_train, y_train, n_epochs=epoch)
        temp, y_pred = model.predict(X_train)

        reversed_int = model.reverse()

        relative, absolute = bit_error([b for b in cipher_text], y_pred)
        mse = mean_squared_error(y_train, temp)

        mse_arr.append(mse)
        rel_err_arr.append(relative)
        abs_err_arr.append(absolute)

    MSE_ARR[epoch] = mse_arr
    REL_ERR_ARR[epoch] = rel_err_arr
    ABS_ERR_ARR[epoch] = abs_err_arr

stop = datetime.now()

print('\n' + '=' * 60 + f'\nDuration: {stop - start}')

# -----------------------------------------------   Save Results   -----------------------------------------------

# res = pd.DataFrame({
#     'mse_10': MSE_ARR[10],
#     'mse_50': MSE_ARR[50],
#     'mse_100': MSE_ARR[100],
#     'mse_200': MSE_ARR[200],
#
#     'relative_error_10': REL_ERR_ARR[10],
#     'relative_error_50': REL_ERR_ARR[50],
#     'relative_error_100': REL_ERR_ARR[100],
#     'relative_error_200': REL_ERR_ARR[200],
#
#     'absolute_error_10': ABS_ERR_ARR[10],
#     'absolute_error_50': ABS_ERR_ARR[50],
#     'absolute_error_100': ABS_ERR_ARR[100],
#     'absolute_error_200': ABS_ERR_ARR[200],
# })
#
# res.to_csv('experiments.csv')

#%%
#####################################################################
''' Correlation between absolute bit error and mean squared error '''
#####################################################################

from scipy.stats import pearsonr
print(pearsonr(MSE_ARR[200], REL_ERR_ARR[200]))

import seaborn as sns
sns.regplot(x=MSE_ARR[200], y=REL_ERR_ARR[200])
plt.xlabel('Hamming Distance (Absolute Bit Error)')
plt.ylabel('MSE')
plt.show()


#%%
##############################
''' Deciphering error rate '''
##############################
hidden = []
act = []
err = []

for n_hidden in range(1, 7):
    for i, _ in enumerate(ACTIVATION):

        mistakes = 0
        for j in tqdm(range(200)):

            model = Model11(activation_func=ACTIVATION[i], n_hidden=n_hidden)
            model.compile()

            history = model.fit(X_train, y_train, n_epochs=10)

            try:
                temp, y_pred = model.predict(X_train)
                reversed_int = model.reverse(INVERSES[i])

                true = np.array([b for b in text.encode('cp437')])
                pred = np.array(reversed_int)

                if not np.array_equal(true, pred):
                    mistakes += 1

            except ValueError:
                mistakes += 1
                continue

        hidden.append(n_hidden)
        act.append(NAMES[i])
        err.append(mistakes / 200)


df2 = pd.DataFrame({
    'Num hidden layers': hidden,
    'Activation function': act,
    'Decryption error rate': err
})
# df2.to_csv('df.csv')
