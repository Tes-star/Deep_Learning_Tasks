import keras
from imblearn.over_sampling import SMOTE
import pandas as pd
from keras import Sequential, initializers
from keras.activations import sigmoid, tanh, elu, selu, relu
from keras.initializers.initializers_v2 import GlorotNormal, HeNormal
from keras.utils import to_categorical
from numpy import mean
from sklearn import model_selection
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.utils import shuffle
from tensorflow.python.ops.init_ops_v2 import lecun_normal

import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.metrics import Precision, Recall
from keras.layers import Dense, Input, Flatten, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, RepeatVector, \
    GRU, RNN, SimpleRNN, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def load_data():
    train = pd.read_csv('../data/01_train/train.csv', header=None)
    sampleTest = pd.read_csv('../data/01_train/sampleTest.csv')
    sampleSubmission = pd.read_csv('../data/01_train/sampleSubmission.csv')
    return train, sampleTest, sampleSubmission


def create_traindataset(data, reduce):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    n_future = 7  # next 4 days temperature forecast
    n_past = 90  # Past 90 days

    wandb.log({'random_state': 12345})

    data = data.sample(frac=1, random_state=12345).reset_index(drop=True)
    train = data.to_numpy()[:14]  #
    test = data.to_numpy()[14:]
    wandb.log({'random_state': train.shape})
    wandb.log({'random_state': test.shape})

    for j in range(0, len(train)):
        for i in range(0, len(train[j]) - n_past - n_future + 1):
            x_train.append(train[j, i: i + n_past])
            y_train.append(train[j, i + n_past: i + n_past + n_future])

    for j in range(0, len(test)):
        for i in range(0, len(test[j]) - n_past - n_future + 1):
            x_test.append(test[j, i: i + n_past])
            y_test.append(test[j, i + n_past: i + n_past + n_future])

    def reduce_data(data):
        data_return = []
        for x in data:
            datapoint = np.zeros(30)
            for week in range(10):
                datapoint[week] = mean(x[week * 7:week * 7 + 6])
            for last_days in range(1, 21):
                datapoint[9 + last_days] = x[69 + last_days]
            data_return.append(datapoint)
        return data_return

    if (reduce == 'TRUE'):
        x_train = reduce_data(x_train)
        x_test = reduce_data(x_test)

    x_train = np.expand_dims(np.array(x_train), axis=-1)
    y_train = np.expand_dims(np.array(y_train), axis=-1)

    x_test = np.expand_dims(np.array(x_test), axis=-1)
    y_test = np.expand_dims(np.array(y_test), axis=-1)

    # x_train, y_train = shuffle(x_train, y_train)
    # x_test, y_test = shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test


def load_model(x_train, lstm_units, lstm_size, dropout_rate, activation_lstm_loop,
               activation_lstm_classifier, activation_lstm_classifier_init, activation_lstm_loop_init, rnn_cell,
               end_dense, start_dense, batch_normalisation):
    rnn_cell = 'iRNN'
    match rnn_cell:
        case 'LSTM':
            component_loop = LSTM(units=lstm_units, return_sequences=True, activation=activation_lstm_classifier,
                                  kernel_initializer=activation_lstm_classifier_init
                                  )
        case 'GRU':
            component_loop = GRU(units=lstm_units, return_sequences=True, activation=activation_lstm_classifier,
                                 kernel_initializer=activation_lstm_classifier_init
                                 )
        case 'RNN':
            component_loop = SimpleRNN(units=lstm_units, return_sequences=True, activation=activation_lstm_classifier,
                                       kernel_initializer=activation_lstm_classifier_init
                                       )
        case 'iRNN':
            component_loop = SimpleRNN(units=lstm_units,
                                       kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                       recurrent_initializer=initializers.Identity(gain=1.0),
                                       activation=activation_lstm_classifier)
    # model = Sequential()
    input = Input(shape=(x_train.shape[1], 1))
    norm1 = BatchNormalization()(input)

    irnn = component_loop(norm1)

    if end_dense == 'TRUE':
        norm2 = BatchNormalization()(irnn)
        drop3 = Dropout(dropout_rate)(norm2)
        dens = Dense(units=lstm_units, activation=activation_lstm_loop,
                     kernel_initializer=activation_lstm_loop_init)(drop3)
        norm3 = BatchNormalization()(dens)
        drop4 = Dropout(dropout_rate)(norm3)
    else:
        norm2 = BatchNormalization()(irnn)
        drop4 = Dropout(dropout_rate)(norm2)

    output = Dense(units=7, activation=activation_lstm_classifier, kernel_initializer=activation_lstm_classifier_init)(
        drop4)

    model = keras.Model(input, output)
    # model.summary()
    return model


def evaluate_model(model, x_test, y_test, x_train, y_train, scaler):
    x = tf.concat([x_train, x_test], 0)
    y_pred = model.predict(x)
    # y_pred= tf.convert_to_tensor(np.expand_dims(y_pred, axis=-1))

    y = tf.concat([y_train, y_test], 0)

    mse = mean_squared_error(y_pred, np.squeeze(y.numpy()))
    wandb.log({'mse_dataset': mse})


def train_model():
    wandb.init(reinit=True)
    # wandb.init(group="experiment_1", job_type="eval")
    # wandb.init(group=wandb.util.generate_id(), job_type="split_" + str(fold))
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    train, sampleTest, sampleSubmission = load_data()
    x_train, y_train, x_test, y_test = create_traindataset(data=train, reduce=config.reduce)

    # initiate the k-fold class from model_selection module
    splits = 2
    kf = model_selection.KFold(splits, shuffle=True)
    # for fold, (trn_, val_) in enumerate(kf.split(X=x_train)):
    # set wandb configs

    # set configs

    # neurons = config.neurons
    # optimizer
    match config.optimizer:
        case 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        case 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
        case 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
        case 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=config.learning_rate)
        case 'Adamax':
            optimizer = tf.keras.optimizers.Adamax(learning_rate=config.learning_rate)
        case 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=config.learning_rate)
        case 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=config.learning_rate)
        case 'Ftrl':
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=config.learning_rate)

    # Scale Data
    match config.scaler:
        case 'None':
            scaler = None

            # if no scaler
            baseline1 = 100
            baseline2 = 40
            baseline3 = 15
            baseline4 = 11
        case 'StandardScaler':
            scaler = StandardScaler()
            shape = x_train.shape

            scaler.fit_transform(x_train.reshape(-1, 1))

            x_train = scaler.transform(x_train.reshape(-1, 1))
            x_train = x_train.reshape(shape)

            shape = x_test.shape
            x_test = scaler.transform(x_test.reshape(-1, 1))
            x_test = x_test.reshape(shape)
            # baseline1 = 1
            # baseline2 = 0.8
            # baseline3 = 0.5
            # baseline4 = 0.2

            # if no scaler
            baseline1 = 30
            baseline2 = 13
            baseline3 = 12
            baseline4 = 11
        case 'MinMaxScaler':
            scaler = MinMaxScaler()
            baseline1 = 0.035
            baseline2 = 0.015
            baseline3 = 0.013
            baseline4 = 0.013

    match config.activation_lstm_loop:
        case 'selu':
            activation_lstm_loop = selu
            activation_lstm_loop_init = GlorotNormal
        case 'elu':
            activation_lstm_loop = elu
            activation_lstm_loop_init = GlorotNormal
        case 'tanh':
            activation_lstm_loop = tanh
            activation_lstm_loop_init = GlorotNormal
        case 'sigmoid':
            activation_lstm_loop = sigmoid
            activation_lstm_loop_init = GlorotNormal
        case 'relu':
            activation_lstm_loop = relu
            activation_lstm_loop_init = GlorotNormal

    match config.activation_lstm_classifier:
        case 'selu':
            activation_lstm_classifier = selu
            activation_lstm_classifier_init = GlorotNormal
        case 'elu':
            activation_lstm_classifier = elu
            activation_lstm_classifier_init = GlorotNormal
        case 'sigmoid':
            activation_lstm_classifier = sigmoid
            activation_lstm_classifier_init = GlorotNormal
        case 'relu':
            activation_lstm_classifier = relu
            activation_lstm_classifier_init = GlorotNormal
        case 'tanh':
            activation_lstm_classifier = tanh
            activation_lstm_classifier_init = GlorotNormal
        case 'linear':
            activation_lstm_classifier = 'linear'
            activation_lstm_classifier_init = GlorotNormal
    if config.kernel_init == 'FALSE':
        activation_lstm_classifier_init = None
        activation_lstm_loop_init = None

    # scaler.fit_transform(x.reshape(-1, 1))
    #
    # y = scaler.transform(y.reshape(-1, 1))
    # y = y.reshape(63972, 7)
    #
    # x = scaler.transform(x.reshape(-1, 1))
    # x = x.reshape(63972, 90, 1)

    # )
    # split dataset

    dataset_size = int(x_train.shape[0] * config.data_proportion)
    wandb.log({'dataset_size': dataset_size})
    if dataset_size > 0 and config.reduce == 'False':
        x_train = x_train[:dataset_size]
        y_train = y_train[:dataset_size]
        x_test = x_test[:dataset_size]
        y_test = y_test[:dataset_size]

    early_stopping = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0.00,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    early_stopping_baseline1 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline1
    )
    early_stopping_baseline2 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=40,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline2
    )
    early_stopping_baseline3 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=50,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline3
    )
    early_stopping_baseline4 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=125,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline4
    )
    TerminateOnNaN = keras.callbacks.TerminateOnNaN()

    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    model = load_model(x_train=x_train, lstm_units=config.lstm_units, lstm_size=config.LSTM_size,
                       dropout_rate=config.dropout_rate,
                       activation_lstm_loop=activation_lstm_loop,
                       activation_lstm_loop_init=activation_lstm_loop_init,
                       activation_lstm_classifier=activation_lstm_classifier,
                       activation_lstm_classifier_init=activation_lstm_classifier_init,
                       rnn_cell=config.rnn_cell,
                       start_dense=config.start_dense,
                       end_dense=config.end_dense,
                       batch_normalisation=config.batch_normalisation
                       )
    print(y_test.shape)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.summary()

    model.fit(x_train, y_train, epochs=1000, batch_size=config.batch_size, verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[WandbCallback(), early_stopping, early_stopping_baseline1, early_stopping_baseline2,early_stopping_baseline3,early_stopping_baseline4,
                         TerminateOnNaN]
              )
    #
    evaluate_model(model, x_test, y_test, x_train, y_train, scaler)

    print("Finshed Job")
    wandb.finish()
    print('break_start')

    # print('break_did_not_start')


if __name__ == '__main__':
    """
    Better use Sweep_upload_data.ipynb to avoid errors and bad visualisation
    """
    # load data

    # define sweep_id
    sweep_id = 'j0ig0vlx'
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Abgabe_02",
                entity="deep_learning_hsa")
