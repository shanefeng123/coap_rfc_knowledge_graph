import numpy as np
from itertools import combinations, permutations
from transformers import TFAutoModel, BertTokenizer
import tensorflow as tf
from collections import Counter
import random

# Load the annotated sentences from entity extraction
sentence_tokens = np.load("data/sentence_tokens.npy")
print(sentence_tokens.shape)
y = np.load("data/entity_labels.npy")
print(y.shape)
sentence_tokens = list(sentence_tokens.tolist())
y = list(y.tolist())


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


def add_entity_marks(sentence, labels):
    # TODO: Check if there is a bug when there is no entity in the sentence. Also, are the markers added correctly??
    """
    Args:
        sentence: The tokenized input sentence
        labels: Entity extraction labels for a sentence

    Returns: A sentence with entity marks added

    """
    processed_sentences = []
    entities_positions = extract_entities_positions(labels)
    positions_combinations = combinations(entities_positions, 2)
    for com in positions_combinations:
        first_entity_positions = com[0]
        second_entity_positions = com[1]
        first_entity_positions = (first_entity_positions[0], first_entity_positions[1] + 2)
        if second_entity_positions[0] > first_entity_positions[0]:
            second_entity_positions = (second_entity_positions[0] + 2, second_entity_positions[1] + 4)
        else:
            second_entity_positions = (second_entity_positions[0], second_entity_positions[1] + 2)
        processed_sentence = sentence.copy()
        processed_sentence.insert(first_entity_positions[0], "[E1]")
        processed_sentence.insert(first_entity_positions[1], "[/E1]")
        processed_sentence.insert(second_entity_positions[0], "[E2]")
        processed_sentence.insert(second_entity_positions[1], "[/E2]")
        processed_sentences.append(processed_sentence)
    return processed_sentences


def extract_entities_pair(processed_sentence):
    """
    Args:
        processed_sentence: A sentence with entity marks added

    Returns: entity pair in the sentence

    """
    e1_start = processed_sentence.index("[E1]")
    e1_end = processed_sentence.index("[/E1]")
    entity_1 = processed_sentence[e1_start + 1: e1_end]
    e2_start = processed_sentence.index("[E2]")
    e2_end = processed_sentence.index("[/E2]")
    entity_2 = processed_sentence[e2_start + 1: e2_end]
    return [entity_1, entity_2]


# Use the above functions to process the sentence data
processed_sentences = []
for i in range(len(sentence_tokens)):
    processed_sentences.append(add_entity_marks(sentence_tokens[i], y[i]))

squeeze_processed_sentences = []
for sentences in processed_sentences:
    squeeze_processed_sentences.extend(sentences)

sentence_entities_pairs = []
for sentence in squeeze_processed_sentences:
    sentence_entities_pairs.append(extract_entities_pair(sentence))

# Class of relations there might be. This list will grow if new relations are encountered
RELATIONS = ["NO RELATION", "EQUIVALENT", "APPLICATION", "COMPONENT", "TYPE", "ARCHITECTURE", "OP-TYPE", "OP-FEATURE",
             "OPERATION", "OP-RESULT", "ORIGIN", "PROTOCOL", "REQUIREMENT", "FEATURE", "OP-HOST", "MIDDLEWARE",
             "OP-MIDDLEWARE", "HOST"]
#
# Annotating the relation
relation_labels = []
# 0
relation_labels.append(1)
relation_labels.append(2)
relation_labels.append(3)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(2)
relation_labels.append(2)
relation_labels.append(3)
relation_labels.append(0)
# 10
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(2)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 20
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(4)
relation_labels.append(5)
relation_labels.append(4)
relation_labels.append(5)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(4)
# 30
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 40
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(5)
relation_labels.append(0)
relation_labels.append(6)
# 50
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 60
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 70
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(8)
relation_labels.append(9)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(10)
# 80
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
# 90
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(11)
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 100
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 110
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(4)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(2)
relation_labels.append(0)
# 120
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 130
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 140
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 150
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
# 160
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(0)
# 170
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(11)
relation_labels.append(0)
relation_labels.append(0)
# 180
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 190
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 200
relation_labels.append(12)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 210
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(7)
# 220
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
# 230
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 240
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 250
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(13)
relation_labels.append(0)
# 260
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 270
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 280
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 290
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 300
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(3)
relation_labels.append(3)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
# 310
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 320
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 330
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(14)
relation_labels.append(14)
relation_labels.append(4)
relation_labels.append(15)
relation_labels.append(15)
# 340
relation_labels.append(15)
relation_labels.append(1)
relation_labels.append(4)
relation_labels.append(4)
relation_labels.append(4)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
# 350
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(1)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(7)
relation_labels.append(7)
# 360
relation_labels.append(7)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 370
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(13)
# 380
relation_labels.append(13)
relation_labels.append(13)
relation_labels.append(4)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
# 390
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 400
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(0)
# 410
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 420
relation_labels.append(1)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 430
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(0)
# 440
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
# 450
relation_labels.append(0)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(16)
relation_labels.append(16)
relation_labels.append(8)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
relation_labels.append(15)
# 460
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(17)
relation_labels.append(4)
relation_labels.append(15)
# 470
relation_labels.append(15)
relation_labels.append(2)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(0)
relation_labels.append(15)
relation_labels.append(15)
relation_labels.append(1)
# 480
relation_labels.append(2)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(2)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(0)
relation_labels.append(1)
# 490
relation_labels.append(1)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(6)
relation_labels.append(1)
relation_labels.append(4)
relation_labels.append(0)
relation_labels.append(6)
relation_labels.append(6)
# 500




# Only sample some 0 class to balance out the dataset
num_no_relation = 30
no_relation_indexes = []
y = []
x = []
for i in range(len(relation_labels)):
    if relation_labels[i] == 0:
        no_relation_indexes.append(i)
    else:
        y.append(relation_labels[i])
        x.append(squeeze_processed_sentences[i])

no_relation_indexes = random.choices(no_relation_indexes, k=num_no_relation)

for index in no_relation_indexes:
    x.append(squeeze_processed_sentences[index])
y.extend([0] * num_no_relation)

max_length = len(x[0])
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"], special_tokens=True)
bert = TFAutoModel.from_pretrained(checkpoint)
print(bert.bert.embeddings.vocab_size)
bert.resize_token_embeddings(len(tokenizer))
print(bert.bert.embeddings.vocab_size)
attention_masks = []
for sentence in x:
    pad_index = sentence.index("[PAD]")
    mask = [1] * pad_index
    mask.extend([0] * (max_length - pad_index))
    attention_masks.append(mask)
#
inputs = tf.keras.layers.Input(shape=(max_length,), name="input_ids", dtype="int32")
mask = tf.keras.layers.Input(shape=(max_length,), name="attention_mask", dtype="int32")
embeddings = bert(input_ids=inputs, attention_mask=mask)[0]
CLS_embeddings = embeddings[:, 0]
X = tf.keras.layers.Dense(128, activation="relu")(CLS_embeddings)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(64, activation="relu")(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(32, activation="relu")(X)
X = tf.keras.layers.Dropout(0.2)(X)
outputs = tf.keras.layers.Dense(len(RELATIONS), activation="softmax", name="output")(X)
model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs)
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", precision, recall])
model.summary()
#
input_ids = []
for sentence in x:
    ids = []
    for word in sentence:
        if word in ["[E1]", "[/E1]", "[E2]", "[/E2]"]:
            ids.append(tokenizer(word)["input_ids"][1])
        else:
            ids.append(tokenizer.vocab[word])
    input_ids.append(ids)

x = np.array(input_ids)
attention_masks = np.array(attention_masks)
# x = np.concatenate((x, x_mask), axis=1)

y = np.array(y)
y_one_hot = []
for i in range(len(y)):
    one_hot = [0] * len(RELATIONS)
    one_hot[y[i]] = 1
    y_one_hot.append(one_hot)
y_one_hot = np.array(y_one_hot)

model.fit([x, attention_masks], y_one_hot, batch_size=32, epochs=10)
print(np.argmax(model.predict(x=[x, attention_masks]), axis=-1))
print(Counter(relation_labels))
