from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import create_optimizer

# %% md
# Load CoNLL 2003 Dataset
# ![](images/conll2003.png)
# %%
conll = load_dataset('conll2003')
conll
# %% md
# Daten liegen im CoNLL-Format vor und enthalten neben Tokens und Named Entities auch POS Tags und Phrasenannotationen.
# ![](images/conll-format.png)
# %% md
## Tokens des ersten Trainingsatzes
# %%
conll['train'][0]['tokens']
# %% md
## POS Tags des ersten Trainingssatzes
# Es wird das [Penn Treebank Tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) genutzt.
# %%
conll['train'][0]['pos_tags']
# %%
conll['train'].features['pos_tags']
# %% md
# Es wird das BIO-Encoding-Schema genutzt.
# %%
conll['train'][0]['ner_tags']
# %%
conll['train'].features['ner_tags']
# %% md
# Data preprocessing
# Wir nutzen das Language Model [distilbert-base-cased](https://huggingface.co/distilbert-base-cased).
# %%
model_name = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %% md
# Language Model Tokenizer tokenisieren Text in Tokens und Subtokens. Falls der zu verarbeitende Text bereits tokenisiert vorliegt, dann sollte diese Tokenisierung für NER beibehalten werden. Eine andere Tokenisierung würde dazu führen, dass Tokengrenzen nicht mehr mit den Named Entity Labels übereinstimmen.
# Ein erneutes Tokenisieren verhindern wir durch den Parameter ```is_split_into_words=True```
# %%
inputs = tokenizer(conll['train'][0]['tokens'], is_split_into_words=True)
inputs.tokens()


# %% md
# Named Entity labels müssen korrigiert werden: nur das erste Subtoken je Token wird mit einem Label versehen, weitere Subtokens werden ignoriert ([https://huggingface.co/docs/transformers/tasks/token_classification](https://huggingface.co/docs/transformers/tasks/token_classification)).
# %%
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
            # if the first subtoken of the next token is encountered, then associate the token's ner label with the subtoken
            # FIXME what if two consecutive tokens are identical (e.g. "is this a really really bad?")?
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # consecutive subtokens will be ignored
            else:
                label_ids.append(-100)
            # memorize the current word
            previous_word_idx = word_idx
        # add labels of the current sequence to the list of labels of the batch
        labels.append(label_ids)

    # update batch labels
    tokenized_sequences['labels'] = labels
    return tokenized_sequences


# %% md
# Model and training definition
# Load auto model for token classification. This adds a classification layer / head to the language model (usually a simple dense layer and dropout).
# %%
model = TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)
# %% md
# Specify training parameters and optimizer.
# %%
batch_size = 1
num_train_epochs = 1
num_train_steps = (len(conll['train']) // batch_size) * num_train_epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=2e-5,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
    num_warmup_steps=0,
)

model.get_layer('distilbert').trainable = False

model.compile(optimizer=optimizer, metrics='acc')
model.summary()
# %% md
# %%
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
tf_train_set = model.prepare_tf_dataset(
    conll['train'].map(tokenize_and_align_labels, batched=True),
    shuffle=True,
    batch_size=1,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    conll['validation'].map(tokenize_and_align_labels, batched=True),
    shuffle=False,
    batch_size=1,
    collate_fn=data_collator,
)
# %% md
# Train model
# %%
import os
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=C:/Users/Timo/PycharmProjects/shared-tasks-wintersemester-2022-23" \
                        "/nvvm/libdevice "
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_train_epochs)
