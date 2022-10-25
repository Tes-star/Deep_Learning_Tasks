from imblearn.over_sampling import SMOTE
import pandas as pd
from keras import Sequential
from keras.activations import sigmoid, tanh, elu, selu
from keras.initializers.initializers_v2 import GlorotNormal, HeNormal
from keras.utils import to_categorical
from sklearn import model_selection
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from tensorflow.python.ops.init_ops import lecun_normal

import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.metrics import Precision, Recall
from keras.layers import Dense, Input, Flatten, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, RepeatVector, \
    GRU, RNN, SimpleRNN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'mean_squared_error'},
    'parameters': {
        'batch_size': {
            'values':
                [5000]
        },
        'dropout_rate': {
            'values':
                [0.2, 0.3, 0.4, 0.5]
        },
        # 'dropout_rate_change': {
        #     'values':
        #         [0.9, 1, 1.1]
        # },
        'learning_rate': {
            'values':
                [0.01, 0.001, 0.0001]
        },
        # 'loss': {
        #     'values':
        #         ['CategoricalCrossentropy']
        # },

        # 'neurons': {'max': 2000, 'min': 200},
        'lstm_units': {'max': 5, 'min': 2},
        'LSTM_size': {'max': 60, 'min': 30},

        # 'neurons_rate_change': {
        #     'values':
        #         [0.5, 0.75, 0.9, 1]
        # },
        # 'num_weights': {
        #    'values':
        #        [200000, 400000, 800000]
        # },
        'optimizer': {
            'values':
                ['Adam', 'RMSprop', 'Adadelta', 'Adamax', 'Nadam', 'Adagrad']
        },
        'activation_lstm_loop': {
            'values':
                ['elu', 'tanh', 'sigmoid']
        },
        'activation_lstm_classifier': {
            'values':
                ['elu', 'tanh', 'sigmoid']
        },
        'scaler': {
            'values':
                ['StandardScaler', 'MinMaxScaler']
        },
        'lstm_Bidirectional': {
            'values':
                ['TRUE', 'FALSE']  # ,
        }
    },
    'program': 'model.py'
}


def load_data():
    train = pd.read_csv('../data/01_train/train.csv', header=None)
    sampleTest = pd.read_csv('../data/01_train/sampleTest.csv')
    sampleSubmission = pd.read_csv('../data/01_train/sampleSubmission.csv')
    return train, sampleTest, sampleSubmission


def create_traindataset(train):
    x = []
    y = []
    n_future = 7  # next 4 days temperature forecast
    n_past = 90  # Past 90 days

    train = train.to_numpy()
    for j in range(0, len(train)):
        for i in range(0, len(train[j]) - n_past - n_future + 1):
            x.append(train[j, i: i + n_past])
            y.append(train[j, i + n_past: i + n_past + n_future])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def load_model(x_train, lstm_units, lstm_size, dropout_rate, lstm_Bidirectional, activation_lstm_loop,
               activation_lstm_classifier, activation_lstm_classifier_init, activation_lstm_loop_init, rnn_cell):
    match rnn_cell:
        case 'LSTM':
            component_first = LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1),
                                   activation=activation_lstm_loop, kernel_initializer=activation_lstm_loop_init)
            component_loop = LSTM(units=lstm_units, return_sequences=True, activation=activation_lstm_loop,
                                  kernel_initializer=activation_lstm_loop_init)
            component_end = LSTM(units=lstm_units)
        case 'GRU':
            component_first = GRU(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1),
                                  activation=activation_lstm_loop, kernel_initializer=activation_lstm_loop_init)
            component_loop = GRU(units=lstm_units, return_sequences=True, activation=activation_lstm_loop,
                                 kernel_initializer=activation_lstm_loop_init)
            component_end = GRU(units=lstm_units)
        case 'RNN':
            component_end = SimpleRNN(units=lstm_units)
            component_first = SimpleRNN(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1),
                                        activation=activation_lstm_loop)
            component_loop = SimpleRNN(units=lstm_units, return_sequences=True, activation=activation_lstm_loop)

    model = Sequential()
    model.add(Bidirectional(component_first))
    model.add(Dropout(dropout_rate))
    for i in range(lstm_size):
        if lstm_Bidirectional == 'True':
            model.add(Bidirectional(component_loop))
        else:
            model.add(component_loop)
        model.add(Dropout(dropout_rate))
    model.add(component_end)
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=7, activation=activation_lstm_classifier, kernel_initializer=activation_lstm_classifier_init))
    # model.build(x_train.shape[1], 1)
    # model.summary()
    return model


def evaluate_model(model, x_test, y_test, scaler):
    y_pred = model.predict(x_test)
    print('x_test scaled:')
    print(x_test[1,:20].reshape(1,-1))
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
    train, sampleTest, sampleSubmission = load_data()
    x, y = create_traindataset(train)

    # initiate the k-fold class from model_selection module
    splits = 2
    kf = model_selection.KFold(splits, shuffle=True)
    for fold, (trn_, val_) in enumerate(kf.split(X=x)):
        # set wandb configs
        wandb.init(reinit=True)
        # wandb.init(group="experiment_1", job_type="eval")
        # wandb.init(group=wandb.util.generate_id(), job_type="split_" + str(fold))
        # Access all hyperparameter values through wandb.config
        config = wandb.config

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
                baseline1 = 1
                baseline2 = 0.8
                baseline3 = 0.5
                baseline4 = 0.2
            case 'MinMaxScaler':
                scaler = MinMaxScaler()
                baseline1 = 0.035
                baseline2 = 0.015
                baseline3 = 0.013
                baseline4 = 0.013
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
            case 'tanh':
                activation_lstm_classifier = tanh
                activation_lstm_classifier_init = lecun_normal
            case 'linear':
                activation_lstm_classifier = 'linear'
                activation_lstm_classifier_init = GlorotNormal
        if config.kernel_init == 'FALSE':
            activation_lstm_classifier_init = None
            activation_lstm_loop_init = None

        scaler.fit_transform(x.reshape(-1, 1))

        y = scaler.transform(y.reshape(-1, 1))
        y = y.reshape(63972, 7)

        x = scaler.transform(x.reshape(-1, 1))
        x = x.reshape(63972, 90, 1)

        # )
        # split dataset
        dataset_size = int(x.shape[0] * config.data_proportion)
        wandb.log({'dataset_size': dataset_size})
        if dataset_size > 0:
            x_train = x[trn_[:dataset_size]]
            y_train = y[trn_[:dataset_size]]
            x_test = x[val_[:int(dataset_size * 0.3)]]
            y_test = y[val_[:int(dataset_size * 0.3)]]
        else:
            x_train = x[trn_]
            y_train = y[trn_]
            x_test = x[val_]
            y_test = y[val_]

        early_stopping = EarlyStopping(
            monitor='val_mean_squared_error',
            min_delta=0.001,  # minimium amount of change to count as an improvement
            patience=10,  # how many epochs to wait before stopping
            restore_best_weights=True,
        )
        early_stopping_baseline1 = EarlyStopping(
            monitor='val_mean_squared_error',
            min_delta=0,  # minimium amount of change to count as an improvement
            patience=5,  # how many epochs to wait before stopping
            restore_best_weights=True,
            baseline=baseline1
        )
        early_stopping_baseline2 = EarlyStopping(
            monitor='val_mean_squared_error',
            min_delta=0,  # minimium amount of change to count as an improvement
            patience=20,  # how many epochs to wait before stopping
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
        # print(x_train.shape)
        # print(y_train.shape)
        # print(x_test.shape)
        # print(y_test.shape)
        model = load_model(x_train=x_train, lstm_units=config.lstm_units, lstm_size=config.LSTM_size,
                           dropout_rate=config.dropout_rate,
                           lstm_Bidirectional=config.lstm_Bidirectional,
                           activation_lstm_loop=activation_lstm_loop,
                           activation_lstm_classifier=activation_lstm_classifier,
                           activation_lstm_loop_init=activation_lstm_loop_init,
                           activation_lstm_classifier_init=activation_lstm_classifier_init,
                           rnn_cell=config.rnn_cell)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

        model.fit(x_train, y_train, epochs=400, batch_size=config.batch_size, verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopping, early_stopping_baseline1, early_stopping_baseline2, WandbCallback()]
                  )
        evaluate_model(model, x_test, y_test, scaler)

        print("Finshed Job")
        model.summary()
        wandb.finish()
        print('break_start')
        break
        print('break_did_not_start')


if __name__ == '__main__':
    """
    Better use Sweep_upload_data.ipynb to avoid errors and bad visualisation
    """
    # load data


    # define sweep_id
    sweep_id = 'iugoempw'
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Abgabe_02",
                entity="deep_learning_hsa")
