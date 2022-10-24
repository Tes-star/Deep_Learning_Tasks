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
from keras.layers import Dense, Input, Flatten, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, RepeatVector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sweep_configuration = {
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'mean_squared_error'},
    'parameters': {
        'batch_size': {
            'values':
                [1000, 5000]
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
        'loss': {
            'values':
                ['CategoricalCrossentropy']
        },

        # 'neurons': {'max': 2000, 'min': 200},
        'lstm_units': {'max': 20, 'min': 5},
        'LSTM_size': {'max': 20, 'min': 7},

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
                ['RMSprop', 'Adadelta', 'Adamax', 'Nadam', 'Adagrad', 'Adam']
        },
        'activation_lstm_loop': {
            'values':
                ['elu', 'tanh', 'selu', 'sigmoid']
        },
        'activation_lstm_classifier': {
            'values':
                ['elu', 'tanh', 'selu', 'sigmoid']
        },
        'scaler': {
            'values':
                ['StandardScaler', 'MinMaxScaler']
        },
        'lstm_Bidirectional': {
            'values':
                ['TRUE', 'FALSE']
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
               activation_lstm_classifier, activation_lstm_classifier_init, activation_lstm_loop_init):
    # RepeatVector(len_input),
    # TimeDistributed(Dense(hidden_size, activation = elu)),

    model = Sequential()
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1),
                                 activation=activation_lstm_loop, kernel_initializer=activation_lstm_loop_init)))
    model.add(Dropout(dropout_rate))
    for i in range(lstm_size):
        if lstm_Bidirectional == 'True':
            model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True, activation=activation_lstm_loop,
                                         kernel_initializer=activation_lstm_loop_init)))
        else:
            model.add(LSTM(units=lstm_units, return_sequences=True, activation=activation_lstm_loop))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=7, activation=activation_lstm_classifier, kernel_initializer=activation_lstm_classifier_init))
    # model.build(x_train.shape[1], 1)
    # model.summary()
    return model


def evaluate_model(model, x_test, y_test, scaler):
    y_pred = model.predict(x_test)

    y_pred = y_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
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
            case 'MinMaxScaler':
                scaler = MinMaxScaler()

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

        scaler.fit_transform(x.reshape(-1, 1))

        y = scaler.transform(y.reshape(-1, 1))
        y = y.reshape(63972, 7)

        x = scaler.transform(x.reshape(-1, 1))
        x = x.reshape(63972, 90, 1)

        # )
        # split dataset
        dataset_size = 0
        wandb.log({'dataset_size': dataset_size})
        if dataset_size > 0:
            x_train = x[trn_[:dataset_size]]
            y_train = y[trn_[:dataset_size]]
            x_test = x[val_[:dataset_size]]
            y_test = y[val_[:dataset_size]]
        else:
            x_train = x[trn_]
            y_train = y[trn_]
            x_test = x[val_]
            y_test = y[val_]

        early_stopping = EarlyStopping(
            monitor='val_mean_squared_error',
            min_delta=0.001,  # minimium amount of change to count as an improvement
            patience=5,  # how many epochs to wait before stopping
            restore_best_weights=True,
        )
        # early_stopping_baseline1 = EarlyStopping(
        #     monitor='val_accuracy',
        #     min_delta=0.001,  # minimium amount of change to count as an improvement
        #     patience=50,  # how many epochs to wait before stopping
        #     restore_best_weights=True,
        #     baseline=0.65
        # )
        early_stopping_baseline2 = EarlyStopping(
            monitor='val_mean_squared_error',
            min_delta=0,  # minimium amount of change to count as an improvement
            patience=5,  # how many epochs to wait before stopping
            restore_best_weights=True,
            baseline=0.98
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
                           activation_lstm_classifier_init=activation_lstm_classifier_init)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

        model.fit(x_train, y_train, epochs=5, batch_size=config.batch_size, verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[WandbCallback()]
                  )
        evaluate_model(model, x_test, y_test, scaler)

        print("Finshed Job")
        model.summary()
        wandb.finish()
        break

        # callbacks=[early_stopping, early_stopping_baseline1, early_stopping_baseline2, WandbCallback()]

    # from keras.layers import Bidirectional
    #
    # inputs = tf.keras.Input(shape=(x_train.shape[-1],))
    #
    # # create hidden layers
    # num_weights = config.num_weights
    # free_weights = num_weights
    # neurons_rate_change = config.neurons_rate_change
    # dropout_rate = config.dropout_rate
    # neuron_list = [neurons]
    # free_weights = free_weights - neurons * neuron_list[-1]
    #
    # while free_weights > 0:
    #     neurons = int(neurons * neurons_rate_change)
    #     if free_weights - (neurons * neuron_list[-1]) > 0:
    #         free_weights = free_weights - neurons * neuron_list[-1]
    #         neuron_list.append(neurons)
    #     else:
    #         break
    #
    # neuron_list.sort(reverse=True)
    # # create hidden layers dynamic
    # d = {}
    # d['hl0'] = Dense(neuron_list[0], activation='relu')(inputs)
    # d['hl1'] = Dropout(rate=dropout_rate)(d['hl0'])
    #
    # for i in range(len(neuron_list) - 1):
    #     dropout_rate = min[dropout_rate * config.dropout_rate_change, 0.5]
    #     d["hl{0}".format(i * 2 + 2)] = Dense(neuron_list[1 + i], activation='relu')(d["hl{0}".format(i * 2 + 1)])
    #     d["hl{0}".format(i * 2 + 3)] = Dropout(rate=dropout_rate)(d["hl{0}".format(i * 2 + 2)])
    #
    # # create output neurons
    # output = Dense(10, activation='softmax')(d["hl{0}".format(len(neuron_list) * 2 - 1)])
    #
    # # define model
    # model = Model(inputs=inputs, outputs=output)
    #
    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
    #               optimizer=optimizer,
    #               metrics=['accuracy', Recall(), Precision(), ])
    #
    # model.summary()
    #
    #
    #
    # model.fit(x_train, y_train_oh, epochs=1000, batch_size=config.batch_size, verbose=1
    #           , callbacks=[early_stopping, early_stopping_baseline1, early_stopping_baseline2, WandbCallback()]
    #           , validation_data=(x_val, y_val_oh)
    #           )
    # print('Training ended successful')
    #
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    #
    #
    #

    print("Finshed Job")
    wandb.finish()


if __name__ == '__main__':
    """
    Better use Sweep_upload_data.ipynb to avoid errors and bad visualisation
    """
    # load data

    n_future = 7

    # define sweep_id
    sweep_id = 'gesqt2cs'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Abgabe_02', entity="deep_learning_hsa")
    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Abgabe_02",
                entity="deep_learning_hsa")
