from imblearn.over_sampling import SMOTE
import pandas as pd
from keras import Sequential, initializers
from keras.activations import sigmoid, tanh, elu, selu, relu
from keras.initializers.initializers_v2 import GlorotNormal, HeNormal
from keras.utils import to_categorical
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
    sampleTest = pd.read_csv('../data/01_train/sampleTest.csv', header=None)
    sampleSubmission = pd.read_csv('../data/01_train/sampleSubmission.csv', header=None)
    return train, sampleTest, sampleSubmission


def create_traindataset(data, ):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    n_future = 7  # next 4 days temperature forecast
    n_past = 90  # Past 90 days

    data = data.sample(frac=1, random_state=1234).reset_index(drop=True)
    train = data.to_numpy()[:14]
    test = data.to_numpy()[14:]

    for j in range(0, len(train)):
        for i in range(0, len(train[j]) - n_past - n_future + 1):
            x_train.append(train[j, i: i + n_past])
            y_train.append(train[j, i + n_past: i + n_past + n_future])

    for j in range(0, len(test)):
        for i in range(0, len(test[j]) - n_past - n_future + 1):
            x_test.append(test[j, i: i + n_past])
            y_test.append(test[j, i + n_past: i + n_past + n_future])

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
    match rnn_cell:
        case 'LSTM':
            component_loop = LSTM(units=lstm_units, return_sequences=True, activation=activation_lstm_loop,
                                  kernel_initializer=activation_lstm_loop_init
                                  )
            component_end = LSTM(units=lstm_units, return_sequences=False, activation=activation_lstm_loop,
                                 kernel_initializer=activation_lstm_loop_init
                                 )
        case 'GRU':
            component_loop = GRU(units=lstm_units, return_sequences=True, activation=activation_lstm_loop,
                                 kernel_initializer=activation_lstm_loop_init
                                 )
            component_end = GRU(units=lstm_units, return_sequences=False, activation=activation_lstm_loop,
                                kernel_initializer=activation_lstm_loop_init
                                )
        case 'iRNN':
            component_loop = SimpleRNN(units=lstm_units,
                                       kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                       recurrent_initializer=initializers.Identity(gain=1.0),
                                       activation='relu', return_sequences=True)
            component_end = SimpleRNN(units=lstm_units,
                                      kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                      recurrent_initializer=initializers.Identity(gain=1.0),
                                      activation='relu')
        case 'iGRU':
            component_loop = GRU(units=lstm_units,
                                       kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                       recurrent_initializer=initializers.Identity(gain=1.0),
                                       activation='relu', return_sequences=True)
            component_end = GRU(units=lstm_units,
                                      kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                      recurrent_initializer=initializers.Identity(gain=1.0),
                                      activation='relu')
        case 'iLSTM':
            component_loop = LSTM(units=lstm_units,
                                       kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                       recurrent_initializer=initializers.Identity(gain=1.0),
                                       activation='relu', return_sequences=True)
            component_end = LSTM(units=lstm_units,
                                      kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                      recurrent_initializer=initializers.Identity(gain=1.0),
                                      activation='relu')

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    if batch_normalisation == 'TRUE':
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    if start_dense == 'TRUE':
        model.add(Dense(units=4, activation=activation_lstm_loop,
                        kernel_initializer=activation_lstm_loop_init))
        if batch_normalisation == 'TRUE':
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    for i in range(lstm_size):
        model.add(Bidirectional(component_loop))
        if batch_normalisation == 'TRUE':
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(component_end)
    if batch_normalisation == 'TRUE':
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    if end_dense == 'TRUE':
        model.add(Dense(units=lstm_units, activation=activation_lstm_loop,
                        kernel_initializer=activation_lstm_loop_init))
        if batch_normalisation == 'TRUE':
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(units=7, activation=activation_lstm_classifier, kernel_initializer=activation_lstm_classifier_init))

    # model.summary()
    return model


def evaluate_model(model, x_test, y_test, scaler):
    y_pred = model.predict(x_test)
    print('x_test scaled:')
    print(x_test[1, :20].reshape(1, -1))
    print('y_pred: scaled')
    print(y_pred[0:20])
    print('y_pred: scaled')
    print(y_test[0:20])

    y_pred = y_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    print('y_test:')
    print(y_test[0:20])
    print('y_pred:')
    print(y_pred[0:20])
    mse = mean_squared_error(y_pred, y_test)
    wandb.log({'mse_real': mse})


def train_model():
    wandb.init(reinit=True)
    # wandb.init(group="experiment_1", job_type="eval")
    # wandb.init(group=wandb.util.generate_id(), job_type="split_" + str(fold))
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    train, sampleTest, sampleSubmission = load_data()
    x_train, y_train, x_test, y_test = create_traindataset(data=train)

    # initiate the k-fold class from model_selection module
    splits = 2
    kf = model_selection.KFold(splits, shuffle=True)
    # for fold, (trn_, val_) in enumerate(kf.split(X=x_train)):
    # set wandb configs

    batch_size=config.batch_size
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
        case 'StandardScaler':
            scaler = StandardScaler()
            # baseline1 = 1
            # baseline2 = 0.8
            # baseline3 = 0.5
            # baseline4 = 0.2

            # if no scaler
            baseline1 = 100
            baseline2 = 40
            baseline3 = 15
            baseline4 = 12
        case 'MinMaxScaler':
            scaler = MinMaxScaler()
            baseline1 = 0.035
            baseline2 = 0.015
            baseline3 = 0.013
            baseline4 = 0.013

    shape=x_train.shape

    scaler.fit_transform(x_train.reshape(-1, 1))

    x_train = scaler.transform(x_train.reshape(-1, 1))
    x_train = x_train.reshape(shape)

    shape = x_test.shape
    x_test = scaler.transform(x_test.reshape(-1, 1))
    x_test = x_test.reshape(shape)

    match config.activation_lstm_loop:
        case 'selu':
            activation_lstm_loop = selu
            activation_lstm_loop_init = lecun_normal
        case 'elu':
            activation_lstm_loop = elu
            activation_lstm_loop_init = GlorotNormal
        case 'tanh':
            activation_lstm_loop = tanh
            activation_lstm_loop_init = lecun_normal
        case 'sigmoid':
            activation_lstm_loop = sigmoid
            activation_lstm_loop_init = GlorotNormal
        case 'relu':
            activation_lstm_loop = relu
            activation_lstm_loop_init = GlorotNormal

    match config.activation_lstm_classifier:
        case 'selu':
            activation_lstm_classifier = selu
            activation_lstm_classifier_init = lecun_normal
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
            activation_lstm_classifier_init = lecun_normal
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
    callbacks=[]

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.summary()
    data_proportion = config.data_proportion
    if model.count_params() >= 80000:
        data_proportion = 0.5

        baseline1 = 20
        baseline2 = 11
        early_stopping_baseline_sonder = EarlyStopping(
            monitor='val_mean_squared_error',
            min_delta=0,  # minimium amount of change to count as an improvement
            patience=5,  # how many epochs to wait before stopping
            restore_best_weights=True,
            baseline=12
        )
        callbacks.append(early_stopping_baseline_sonder)

    dataset_size = int(x_train.shape[0] * data_proportion)
    wandb.log({'dataset_size': dataset_size})
    if dataset_size > 0:
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
    callbacks.append(early_stopping)
    early_stopping_baseline1 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline1
    )
    callbacks.append(early_stopping_baseline1)

    early_stopping_baseline2 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline2
    )
    callbacks.append(early_stopping_baseline2)

    early_stopping_baseline3 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=50,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline3
    )
    callbacks.append(early_stopping_baseline3)

    early_stopping_baseline4 = EarlyStopping(
        monitor='val_mean_squared_error',
        min_delta=0,  # minimium amount of change to count as an improvement
        patience=125,  # how many epochs to wait before stopping
        restore_best_weights=True,
        baseline=baseline4
    )
    callbacks.append(early_stopping_baseline4)

    model.fit(x_train, y_train, epochs=1000, batch_size=batch_size, verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)
    #
    #evaluate_model(model, x_test, y_test, scaler)

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
    sweep_id = 'm8hvlv5l'
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Abgabe_02",
                entity="deep_learning_hsa")
