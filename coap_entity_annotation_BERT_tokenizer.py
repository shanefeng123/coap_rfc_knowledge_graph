# %%
from collections import Counter
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFAutoModelForTokenClassification, TFAutoModel
import string
import nltk
import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# # Don't worry about this part first. This is to train a customized tokenizer. Might be useful later
# rfcs = ["rfc7252.txt", "rfc7959.txt", "rfc8613.txt", "rfc8974.txt"]
# tokenizer = BertWordPieceTokenizer(
#     clean_text=True,
#     handle_chinese_chars=False,
#     strip_accents=False,
#     lowercase=True
# )
#
# tokenizer.train(files=rfcs, vocab_size=30_000, min_frequency=2,
#                 limit_alphabet=1000, wordpieces_prefix='##',
#                 special_tokens=[
#                     '[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
# import os
# from pathlib import Path
# if not os.path.isdir('./bert-coap'):
#   os.mkdir('./bert-coap')
#
# tokenizer.save_model('./bert-coap', 'bert-coap')

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('./bert-coap/bert-coap-vocab.txt')

# Load the bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
rfc7252 = open("rfc7252.txt").read()

# Use nltk to tokenize sentences
sentence_text = nltk.sent_tokenize(rfc7252, "english")
sentence_text = sentence_text[210:]
# Remove part of the headers
sentence_text = [sentence for sentence in sentence_text if sentence != 'Shelby, et al.']

for i in range(len(sentence_text)):
    # Remove headers of the document
    sentence = sentence_text[i]
    if sentence.startswith("Standards Track"):
        split_position = sentence.find("2014")
        if split_position != -1:
            sentence_text[i] = sentence[split_position + 4:]
        else:
            sentence_text[i] = ""
    sentence_text[i] = sentence_text[i].replace("\n", "")

    # Remove lines contain numbers only
    alpha = any(c.isalpha() for c in sentence_text[i])
    if not alpha:
        sentence_text[i] = ""

    # Remove figures and tables
    if "Figure" in sentence_text[i] and ":" in sentence_text[i]:
        sentence_text[i] = ""
    if "Table" in sentence_text[i] and ":" in sentence_text[i]:
        sentence_text[i] = ""

    # Change to all lower case
    sentence_text[i] = sentence_text[i].lower()

sentence_text = [sentence for sentence in sentence_text if sentence != ""]

# Remove acknowledgement and references
sentence_text = sentence_text[:1282]

# Plot to see the distribution of the lengths of the sentences
tokenized_sentences = tokenizer(sentence_text)["input_ids"]
lengths = []
for sentence in tokenized_sentences:
    lengths.append(len(sentence))
bins = len(set(lengths))
plt.hist(lengths, bins=bins)
plt.show()

# Use bert tokenizer to tokenize each word in each sentence. Only take 100 as the maximum length to avoid too
# many padding tokens
tokenized_sentences = tokenizer(sentence_text, padding="max_length", max_length=100)
input_ids = tokenized_sentences["input_ids"]
attention_mask = tokenized_sentences["attention_mask"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

# annotations
max_length = len(sentence_tokens[0])


def add_trailing_padding(label_list):
    label_list.extend((max_length - len(label_list)) * [3])


LABELS = ["B-entity", "I-entity", "O", "PAD"]
y = []
# 0
y.append(
    [2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 0, 2, 2, 0, 2,
     2])
add_trailing_padding(y[0])
# 1
y.append(
    [2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1,
     1, 1, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[1])
# 2
y.append([2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 0, 2, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[2])
# 3
y.append([2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[3])
# 4
y.append(
    [2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 1,
     1, 2, 0, 1, 2, 0, 2, 2])
add_trailing_padding(y[4])
# 5
y.append(
    [2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1,
     2, 2])
add_trailing_padding(y[5])
# 6
y.append(
    [2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2, 0, 1, 1, 1, 2, 0, 1,
     1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[6])
# 7
y.append(
    [2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2,
     2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[7])
# 8
y.append(
    [2, 0, 0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1,
     2, 0, 1, 1, 2, 2])
add_trailing_padding(y[8])
# 9
y.append([2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[9])
# 10
y.append([2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[10])
# 11
y.append([2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 2])
add_trailing_padding(y[11])
# 12
y.append([2, 2, 2, 0, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[12])
# 13
y.append(
    [2, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0,
     1, 2, 2])
add_trailing_padding(y[13])
# 14
y.append([2, 2, 0, 2, 2, 0, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[14])
# 15
y.append([2] * 76)
add_trailing_padding(y[15])
# 16
y.append([2] * 20)
add_trailing_padding(y[16])
# 17
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2,
     2, 2, 2, 2, 2])
add_trailing_padding(y[17])
# 18
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2])
add_trailing_padding(y[18])
# 19
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[19])
# 20
y.append(
    [2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2,
     2, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[20])
# 21
y.append([2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[21])
# 22
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2])
add_trailing_padding(y[22])
# 23
y.append([2, 0, 2, 0, 1, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[23])
# 24
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2])
add_trailing_padding(y[24])
# 25
y.append([2, 0, 2, 0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[25])
# 26
y.append([2, 0, 2, 0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[26])
# 27
y.append([2, 0, 1, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[27])
# 28
y.append([2, 0, 1, 1, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2])
add_trailing_padding(y[28])
# 29
y.append([2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[29])
# 30
y.append(
    [2, 0, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2])
add_trailing_padding(y[30])
# 31
y.append([2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2])
add_trailing_padding(y[31])
# 32
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[32])
# 33
y.append([2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[33])
# 34
y.append([2, 0, 1, 1, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[34])
# 35
y.append([2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[35])
# 36
y.append([2, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[36])
# 37
y.append(
    [2, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2,
     0, 1, 2, 2])
add_trailing_padding(y[37])
# 38
y.append(
    [2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0,
     2, 2, 0, 2, 2, 2])
add_trailing_padding(y[38])
# 39
y.append([2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[39])
# 40
# y.append()

# Fine Tune BERT
checkpoint = "bert-base-uncased"
bert = TFAutoModelForTokenClassification.from_pretrained(checkpoint)
inputs = tf.keras.layers.Input(shape=(max_length,), name="input_ids", dtype="int32")
mask = tf.keras.layers.Input(shape=(max_length,), name="attention_mask", dtype="int32")
embeddings = bert(input_ids=inputs, attention_mask=mask)[0]
X = tf.keras.layers.Dense(128, activation="relu")(embeddings)
X = tf.keras.layers.Dropout(0.1)(X)
outputs = tf.keras.layers.Dense(4, activation="softmax", name="output")(X)
model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs)
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", precision, recall])
model.summary()
#
# Just trying out
x = np.array(input_ids[0:40])
x_mask = np.array(attention_mask[0:40])

y_distribution = []
for labels in y:
    y_distribution.extend(labels)
print(Counter(y_distribution))

# One hot the labels so that the model can calculate loss and accuracy
for i in range(len(y)):
    for j in range(len(y[i])):
        one_hot = [0] * len(LABELS)
        one_hot[y[i][j] - 1] = 1
        y[i][j] = one_hot

x_train = x[0:30]
x_val = x[30:]
x_train_mask = x_mask[0:30]
x_val_mask = x_mask[30:]

y = np.array(y)
y_train = y[0:30]
y_val = y[30:]

model.fit([x_train, x_train_mask], y_train, validation_data=([x_val, x_val_mask], y_val), batch_size=2, epochs=10)

# test_sentence = 'The use of web services (web APIs) on the Internet has become ' \
#                 'ubiquitous in most applications and depends on the fundamental ' \
#                 'Representational State Transfer [REST] architecture of the Web.'
# tokenized_test_sentence = tokenizer(test_sentence, padding="max_length", max_length=max_length)
# test_sentence_ids = np.array([tokenized_test_sentence["input_ids"]])
# test_sentence_mask = np.array([tokenized_test_sentence["attention_mask"]])
# print(model.predict(x=[test_sentence_ids, test_sentence_mask]))
