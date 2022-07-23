import numpy as np
from transformers import TFAutoModel, BertTokenizer
import tensorflow as tf
import random
from sklearn.utils import shuffle

from itertools import combinations


def extract_entities_positions(labels):
    """
    Args:
        labels: Entity extraction labels for a sentence

    Returns: The position indexes of the entities in the sentence

    """
    entities_positions = []
    entity_position = []
    for i in range(len(labels)):
        if labels[i] == 0 and entity_position:
            entity_position.append(entity_position[0])
            entities_positions.append(entity_position)
            entity_position = [i]
        elif labels[i] == 0 or labels[i] == 1:
            entity_position.append(i)
        else:
            if entity_position:
                entities_positions.append([entity_position[0], entity_position[-1]])
                entity_position = []
    return entities_positions


# Load the annotated sentences from entity extraction
sentence_tokens = np.load("data/sentence_tokens.npy")
print(sentence_tokens.shape)
y = np.load("data/entity_labels.npy")
print(y.shape)
sentence_tokens = list(sentence_tokens.tolist())
y = list(y.tolist())

relation_statements = []
for i in range(len(sentence_tokens)):
    entities_positions = extract_entities_positions(y[i])
    if not entities_positions:
        continue
    positions_combinations = combinations(entities_positions, 2)
    for com in positions_combinations:
        entity_1 = sentence_tokens[i][com[0][0]: com[0][1] + 1]
        entity_2 = sentence_tokens[i][com[1][0]: com[1][1] + 1]
        relation_statement = ((sentence_tokens[i], com[0], com[1]), entity_1, entity_2)
        relation_statements.append(relation_statement)

for i in range(len(relation_statements)):
    sentence = relation_statements[i][0][0].copy()
    entity_1_position = relation_statements[i][0][1]
    entity_2_position = relation_statements[i][0][2]
    entity_1 = relation_statements[i][1]
    entity_2 = relation_statements[i][2]
    entity_1_position = (entity_1_position[0], entity_1_position[1] + 2)
    entity_2_position = (entity_2_position[0] + 2, entity_2_position[1] + 4)
    sentence.insert(entity_1_position[0], "[E1]")
    sentence.insert(entity_1_position[1], "[/E1]")
    sentence.insert(entity_2_position[0], "[E2]")
    sentence.insert(entity_2_position[1], "[/E2]")
    entity_1_position = (entity_1_position[0] + 1, entity_1_position[1] - 1)
    entity_2_position = (entity_2_position[0] + 1, entity_2_position[1] - 1)
    relation_statements[i] = ((sentence, entity_1_position, entity_2_position), entity_1, entity_2)


def introduce_blank(sentence, start, end, blank_prob):
    result = random.random()
    if result <= blank_prob:
        for i in range(start, end + 1):
            sentence[i] = "[BLANK]"


def prepare_pretrain_dataset(relation_statements, blank_prob):
    X = []
    y = []
    relation_statements_combinations = combinations(relation_statements, 2)
    for com in relation_statements_combinations:
        relation_statement_1 = com[0]
        relation_statement_2 = com[1]
        if relation_statement_1[1] == relation_statement_2[1] and relation_statement_1[2] == relation_statement_2[2]:
            y.append(1)
        else:
            y.append(0)
        sentence_1 = relation_statement_1[0][0].copy()
        sentence_2 = relation_statement_2[0][0].copy()
        introduce_blank(sentence_1, relation_statement_1[0][1][0], relation_statement_1[0][1][1], blank_prob=blank_prob)
        introduce_blank(sentence_1, relation_statement_1[0][2][0], relation_statement_1[0][2][1], blank_prob=blank_prob)
        introduce_blank(sentence_2, relation_statement_2[0][1][0], relation_statement_2[0][1][1], blank_prob=blank_prob)
        introduce_blank(sentence_2, relation_statement_2[0][2][0], relation_statement_2[0][2][1], blank_prob=blank_prob)
        sentence_1 = [x for x in sentence_1 if x != "[PAD]"]
        sentence_2.remove("[CLS]")
        sentence_2.remove("[SEP]")
        sentence_2 = [x for x in sentence_2 if x != "[PAD]"]
        sentence_1.extend(sentence_2)
        X.append(sentence_1)
    return X, y


X, y = prepare_pretrain_dataset(relation_statements, blank_prob=0.3)

max_length = 0
for sentence in X:
    if len(sentence) > max_length:
        max_length = len(sentence)

for i in range(len(X)):
    if len(X[i]) < max_length:
        X[i] = X[i] + ["[PAD]"] * (max_length - len(X[i]))

indexes = list(range(0, len(X)))
positive_samples = []
for i in range(len(y)):
    if y[i] == 1:
        positive_samples.append(X[i])
        indexes.remove(i)

negative_samples = []
for index in random.choices(indexes, k=2 * len(positive_samples)):
    negative_samples.append(X[index])
samples = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"], special_tokens=True)
bert = TFAutoModel.from_pretrained(checkpoint)
print(bert.bert.embeddings.vocab_size)
bert.resize_token_embeddings(len(tokenizer))
print(bert.bert.embeddings.vocab_size)
attention_masks = []
for sentence in samples:
    pad_index = sentence.index("[PAD]")
    mask = [1] * pad_index
    mask.extend([0] * (max_length - pad_index))
    attention_masks.append(mask)

inputs = tf.keras.layers.Input(shape=(max_length,), name="input_ids", dtype="int32")
mask = tf.keras.layers.Input(shape=(max_length,), name="attention_mask", dtype="int32")
embeddings = bert.bert(input_ids=inputs, attention_mask=mask)[1]
# CLS_embeddings = embeddings[:, 0]
X = tf.keras.layers.Dense(128, activation="relu")(embeddings)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(64, activation="relu")(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(32, activation="relu")(X)
X = tf.keras.layers.Dropout(0.2)(X)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(X)
model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

input_ids = []
for sentence in samples:
    ids = []
    for word in sentence:
        if word in ["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"]:
            ids.append(tokenizer(word)["input_ids"][1])
        else:
            ids.append(tokenizer.vocab[word])
    input_ids.append(ids)

x = np.array(input_ids)
attention_masks = np.array(attention_masks)
y = np.array(labels)
temp = np.concatenate((x, attention_masks), axis=1)
temp, y = shuffle(temp, y)
x = temp[:, 0:max_length]
attention_masks = temp[:, max_length:]
best_model_callback = tf.keras.callbacks.ModelCheckpoint("./model/BERT_relation_pretrain", save_best_only=True,
                                                         monitor="val_loss", mode="min")
model.fit([x, attention_masks], y, callbacks=[best_model_callback], batch_size=32, epochs=100)