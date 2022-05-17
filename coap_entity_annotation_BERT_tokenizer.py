from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFAutoModelForTokenClassification
import string
import nltk
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

# %Don't worry about this part first. This is to train a customized tokenizer. Might be useful later
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

# Use bert tokenizer to tokenize each word in each sentence
sentence_tokens_ids = tokenizer(sentence_text, padding=True)["input_ids"]
sentence_tokens = []
for ids in sentence_tokens_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

# annotations
max_length = len(sentence_tokens[0])


def add_trailing_padding(label_list):
    label_list.extend((max_length - len(label_list)) * [2])


LABELS = ["B-entity", "I-entity", "O"]
y = []
# 0
y.append(
    [2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 0, 2, 2, 0, 2,
     2])
add_trailing_padding(y[0])
# 1
y.append(
    [2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1,
     1, 1, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1])
add_trailing_padding(y[1])
# 2
y.append([2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 0, 2, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1])
add_trailing_padding(y[2])
# 3
y.append([2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0])
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
y.append([2, 2, 0, 1, 1, 1, 1, 1, 1, 2])
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

# Fine Tune BERT
checkpoint = "bert-base-uncased"
model = TFAutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(LABELS))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Just trying out
x = np.array(sentence_tokens_ids[0:15])

for i in range(len(y)):
    for j in range(len(y[i])):
        one_hot = [0] * len(LABELS)
        one_hot[y[i][j] - 1] = 1
        y[i][j] = one_hot

x_train = x[0:13]
x_val = x[13:]

y = np.array(y)
y_train = y[0:13]
y_val = y[13:]

model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=1, epochs=3)