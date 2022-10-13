import operator
from imblearn.over_sampling import SMOTE
import pandas as pd
import wandb as wandb
from keras.utils import to_categorical

pd.set_option('display.max_rows', None)
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from wandb.integration.keras import WandbCallback

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from tensorflow import keras
from keras.models import Model
from keras.metrics import Precision, Recall
from keras.layers import Dense, Input, Flatten, Dropout, Embedding


def train_model(learning_rate=0.001, batch_size=64000, dropout_rate=0.05,
                neurons_0=64,
                neurons_1=64,
                neurons_2=64
                ):
    # set wandb configs
    wandb.init()
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    # set configs
    optim = tf.keras.optimizers.Adam(config.learning_rate)
    neurons_0 = config.neurons_0
    neurons_1 = config.neurons_1
    neurons_2 = config.neurons_2

    loss = CategoricalCrossentropy()  # to avoid errors
    match config.loss:
        case 'CategoricalCrossentropy':
            loss = CategoricalCrossentropy()

    # Read in the csv data using pandas
    train = pd.read_csv('data/01_train/train.csv')
    sampleTest = pd.read_csv('data/01_train/sampleTest.csv')
    sampleSubmission = pd.read_csv('data/01_train/sampleSubmission.csv')

    y_labels = train['t'].unique()
    n_class = len(y_labels)

    # drop features without variance
    features_to_drop = train.nunique()
    features_to_drop = features_to_drop.loc[features_to_drop.values == 1].index
    # print(features_to_drop)

    # now drop these columns from both the training and the test datasets
    train = train.drop(features_to_drop, axis=1)
    sampleTest = sampleTest.drop(features_to_drop, axis=1)
    train.isnull().values.any()

    x_train = train.iloc[:, :-1]
    y_train = train['t']

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      train_size=0.7,
                                                      test_size=0.3,
                                                      random_state=42,
                                                      shuffle=True)

    # if class is imbalanced then oversampling
    if(config.oversampling==1):
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    # Scale Data
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    #    scaler = StandardScaler() # to avoid errors

    match config.scaler:
        case 'StandardScaler':
            scaler = StandardScaler()
        case 'MinMaxScaler':
            scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    inputs = tf.keras.Input(shape=(X_train.shape[-1],))

    # create hidden layers
    neuron_list = [int(neurons_0), int(neurons_0), int(neurons_0),
                   int(neurons_1), int(neurons_1), int(neurons_1),
                   int(neurons_2), int(neurons_2), int(neurons_2),int(neurons_2)]
    # reduce neurons_x to layers of neurons
    layers = int(config.layers)
    dropout_rate = config.dropout_rate
    layers=1

    for i in range(10 - layers):
        neuron_list.pop()

    neuron_list.sort(reverse=True)
    # create hidden layers dynamic
    d = {}
    d['hl0'] = Dense(neuron_list[0], activation='relu')(inputs)
    d['hl1'] = Dropout(rate=dropout_rate)(d['hl0'])

    for i in range(layers - 1):
        dropout_rate = dropout_rate * config.dropout_rate_change
        d["hl{0}".format(i * 2 + 2)] = Dense(neuron_list[1 + i], activation='relu')(d["hl{0}".format(i * 2 + 1)])
        d["hl{0}".format(i * 2 + 3)] = Dropout(rate=dropout_rate)(d["hl{0}".format(i * 2 + 2)])

    # create output neurons
    output = Dense(10, activation='softmax')(d["hl{0}".format(layers * 2 - 1)])

    # define model

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




    model = Model(inputs=inputs, outputs=output)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=optimizer,
                  metrics=['accuracy',Recall(), Precision(), ])

    model.summary()

    y_train_oh = to_categorical(y_train)
    y_val_oh = to_categorical(y_val)

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    model.fit(X_train, y_train_oh, epochs=10, batch_size=config.batch_size, verbose=1
              , callbacks=[early_stopping]
              , validation_data=(X_val, y_val_oh)
              )
    print('Training ended successful')

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # prediction
    pred_y_test = model.predict(x=X_val)
    pred_y_test = pred_y_test.argmax(axis=1)

    pred_y_train = model.predict(x=X_train)
    pred_y_train = pred_y_train.argmax(axis=1)
    print('Prediction ended successful')

    # y_true = np.array(y_val).argmax(axis=1)

    # create confusions matrix
    # labels = [0, 1, 2, 3, 4,5,6,7,8,9]
    cm = confusion_matrix(y_val, pred_y_test, labels=y_labels.sort())

    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
    disp.plot()
    plt.show()
    print('ConfusionMatrixDisplay ended successful')

    y_test = y_val
    labels_train = np.unique(np.concatenate((y_train, pred_y_train)))
    labels_test = np.unique(np.concatenate((y_test, pred_y_test)))

    # calculate and log metrics
    precision_train = precision_score(y_train, pred_y_train, labels=labels_train, average='micro')
    precision_test = precision_score(y_test, pred_y_test, labels=labels_test, average='micro')

    recall_train = recall_score(y_train, pred_y_train, labels=labels_train, average='micro')
    recall_test = recall_score(y_test, pred_y_test, labels=labels_test, average='micro')

    f1_train = f1_score(y_train, pred_y_train, labels=labels_train, average='micro')
    f1_test = f1_score(y_test, pred_y_test, labels=labels_test, average='micro')
    print('Testscore ended successful')
    #labels_test_str = operator.itemgetter(*labels_test)(labels_train)
    wandb.log(
        {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_test, preds=pred_y_test)})

    wandb.log({'precision_train': precision_train})
    wandb.log({'precision_test': precision_test})

    wandb.log({'recall_train': recall_train})
    wandb.log({'recall_test': recall_test})

    wandb.log({'f1_train': f1_train})
    wandb.log({'f1_test': f1_test})
    print("Finshed Job")
    wandb.finish()


if __name__ == '__main__':
    """
    Better use Sweep_upload_data.ipynb to avoid errors and bad visualisation
    """

    #wandb.login()

    # define sweep_id
    sweep_id = 'crs415m4'
    # wandb sweep sweep.yaml

    # run the sweep
    wandb.agent(sweep_id, function=train_model, project="Deep_Learning", entity="deep_learning_hsa")
