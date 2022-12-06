# import data
import csv

import keras
import numpy as np
import pandas as pd
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from keras.layers import Softmax
from keras.metrics import Precision, Recall
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow_addons.metrics import F1Score
from transformers import DataCollatorForTokenClassification, TFAutoModelForTokenClassification, create_optimizer, \
    AutoTokenizer


def tokenize_and_align_labels(sequence_batch):
    # tokenize pre-tokenized sequences
    # long sequences will be truncated to respect the maximum token length of the language model (usually 512)
    tokenized_sequences = tokenizer(sequence_batch['tokens'], truncation=True, is_split_into_words=True)

    labels = []
    # iterate over pre-tokenized tokens of single sequences
    for i, label in enumerate(sequence_batch['ner_tags']):
        # get associated word ids of the subtokens
        # if a token has multiple subtokens, then each subtoken is associated with the token's word id
        word_ids = tokenized_sequences.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        # iterate over subtokens
        for word_idx in word_ids:
            # special tokens (e.g. [CLS], [SEP]) get label id -100 -> loss function will ignore them
            if word_idx is None:
                label_ids.append(-100)
            # if the first subtoken of the next token is encountered, then associate the token's ner label with the
            # subtoken FIXME what if two consecutive tokens are identical (e.g. "is this a really really bad?")?
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # consecutive subtokens will be ignored
            else:
                # label_ids.append(-100)
                # für Subword tokens immer das I Label
                if (label[word_idx] >= 0):
                    label_ids.append(int(label[word_idx] / 2) * 2 + 1)
            # memorize the current word
            previous_word_idx = word_idx
        # add labels of the current sequence to the list of labels of the batch
        labels.append(label_ids)

    # update batch labels
    tokenized_sequences['labels'] = labels
    return tokenized_sequences


def get_data():
    df_train = pd.read_csv('../data/01_train/train.tsv', sep='\t',
                           header=0,
                           names=["Nr", "Wort", "Typ1", "Typ2"],
                           quoting=csv.QUOTE_NONE,
                           keep_default_na=False,
                           encoding='utf-8')

    df_train = df_train.drop(df_train[df_train['Nr'] == '#'].index)
    df_train = df_train.reset_index(drop=False)

    'Label dict filter'

    outside = 0
    label_dict = {
        'B-LOC': 2,
        'I-LOC': 3,
        'B-PER': 4,
        'I-PER': 5,
        'B-ORG': 6,
        'I-ORG': 7,
        'B-LOCderiv': outside,
        "B-LOCpart": outside,
        'B-ORGderiv': outside,
        'B-ORGpart': outside,
        'B-OTH': outside,
        'B-OTHderiv': outside,
        'B-OTHpart': outside,
        'B-PERderiv': outside,
        'B-PERpart': outside,
        'I-LOCderiv': outside,
        'I-LOCpart': outside,
        'I-ORGpart': outside,
        'I-OTH': outside,
        'I-OTHderiv': outside,
        'I-OTHpart': outside,
        'I-PERderiv': outside,
        'I-PERpart': outside,
        'O': outside
    }

    # label_list = ['B-LOC',
    #               'I-LOC',
    #               'B-PER',
    #               'I-PER',
    #               'B-ORG',
    #               'I-ORG'
    #               'B-LOCderiv'
    #               "B-LOCpart",
    #               'B-ORGderiv',
    #               'B-ORGpart',
    #               'B-OTH',
    #               'B-OTHderiv',
    #               'B-OTHpart',
    #               'B-PERderiv',
    #               'B-PERpart',
    #               'I-LOCderiv',
    #               'I-LOCpart',
    #               'I-ORGpart',
    #               'I-OTH',
    #               'I-OTHderiv',
    #               'I-OTHpart',
    #               'I-PERderiv',
    #               'I-PERpart',
    #               'O'
    #               ]


    tokens = []
    labels = []
    ids = []

    sentence_token = []
    sentence_labels = []

    i = 0

    id = 0
    for j in range(0, len(df_train)):
        if i > 0 and df_train.Nr[j] == '1':
            tokens.append(sentence_token)
            labels.append(sentence_labels)

            sentence_token, sentence_labels = [], []
            ids.append(id)
            id = id + 1

        sentence_token.append(str(df_train.Wort[j]))
        # sentence_labels.append(df_train.Typ1[j])
        sentence_labels.append(label_dict.get(df_train.Typ1[j]))

        i = i + 1
    return tokens, labels, ids


def compute_metrics(p):
    import evaluate
    label_list = tf_train_set.features[f"ner_tags"].feature.names

    seqeval = evaluate.load("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def get_model(model_name):
    'LMtraining'

    # model_name='jplu/tf-xlm-r-ner-40-lang'
    # model_name='Davlan/bert-base-multilingual-cased-ner-hrl'

    transformer_model = TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=8)

    #  output = keras.layers.Dense(9, activation='Softmax')(transformer_model)

    model = transformer_model  # keras.models.Model(inputs=transformer_model, outputs=output)

    model.get_layer('bert').trainable = False

    return model, num_train_epochs, batch_size


def get_preprocessing(model, model_name):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
    tf_train_set = model.prepare_tf_dataset(
        conll['train'].map(tokenize_and_align_labels, batched=True),
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        conll['validation'].map(tokenize_and_align_labels, batched=True),
        shuffle=True,
        batch_size=100,
        collate_fn=data_collator,
    )
    return tf_train_set, tf_validation_set



def configurate_model():

    num_train_steps = (len(conll['train']) // batch_size) * num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
        num_warmup_steps=5,
    )
    model.compile(optimizer=optimizer,  # loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

def train_model():
    from transformers.keras_callbacks import KerasMetricCallback
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

    import os
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_train_epochs,callbacks=metric_callback)


def test_model():
    model.predict()


if __name__ == '__main__':
    batch_size = 300
    num_train_epochs = 100

    tokens, labels, ids = get_data()

    model_name = 'deepset/gbert-base'

    conll = {'train': Dataset.from_dict({'tokens': tokens, 'ner_tags': labels})
        , 'validation': Dataset.from_dict({'tokens': tokens[10:11], 'ner_tags': labels[10:11]})
             # 'test':Dataset.from_dict({'label':y_test,'text':x_test})
             }

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(conll['train'][0]['tokens'], is_split_into_words=True)
    inputs.tokens()

    model, num_train_epochs, batch_size = get_model(model_name=model_name)

    # keras.utils.plot_model(model)
    tf_train_set, tf_validation_set = get_preprocessing(model=model, model_name=model_name)

    configurate_model()

    train_model()
