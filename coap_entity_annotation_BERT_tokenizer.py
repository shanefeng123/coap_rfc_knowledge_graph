# %%
import random
from collections import Counter
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFAutoModel
import string
import nltk
import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
rfc7252 = open("./data/rfc7252.txt").read()

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

for i in range(len(sentence_text)):
    if sentence_text[i].endswith("shelby, et al."):
        sentence_text[i] = sentence_text[i][0:sentence_text[i].find("shelby, et al.")] + sentence_text[i + 1]
        sentence_text[i + 1] = ""

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


LABELS = ["B-entity", "I-entity", "Other", "PAD"]
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
y.append(
    [2, 0, 1, 1, 2, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1,
     2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[40])
# 41
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[41])
# 42
y.append([2, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[42])
# 43
y.append([2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[43])
# 44
y.append([2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[44])
# 45
y.append([2, 0, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[45])
# 46
y.append([2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[46])
# 47
y.append([2, 0, 1, 1, 2, 0, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2])
add_trailing_padding(y[47])
# 48
y.append(
    [2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 1,
     1, 1, 1, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[48])
# 49
y.append([2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[49])
# 50
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[50])
# 51
y.append(
    [2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0,
     1, 1, 2, 2, 2, 2])
add_trailing_padding(y[51])
# 52
y.append(
    [2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[52])
# 53
y.append(
    [2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
     1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[53])
# 54
y.append([2, 0, 1, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2])
add_trailing_padding(y[54])
# 55
y.append([2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[55])
# 56
y.append([2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[56])
# 57
y.append(
    [2, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 0,
     2, 2])
add_trailing_padding(y[57])
# 58
y.append([2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[58])
# 59
y.append([2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[59])
# 60
y.append([2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[60])
# 61
y.append([2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[61])
# 62
y.append([2, 0, 1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[62])
# 63
y.append([2, 0, 1, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[63])
# 64
y.append([2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[64])
# 65
y.append(
    [2, 0, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0,
     1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 0, 2, 2])
add_trailing_padding(y[65])
# 66
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[66])
# 67
y.append([2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[67])
# 68
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[68])
# 69
y.append([2, 2, 0, 1, 1, 1, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[69])
# 70
y.append([2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[70])
# 71
y.append([2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 2])
add_trailing_padding(y[71])
# 72
y.append([2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 0, 2, 2, 2])
add_trailing_padding(y[72])
# 73
y.append(
    [2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2,
     2, 0, 2, 2])
add_trailing_padding(y[73])
# 74
y.append([2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[74])
# 75
y.append([2, 2, 0, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[75])
# 76
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2])
add_trailing_padding(y[76])
# 77
y.append([2, 0, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1, 1, 2, 0, 1, 2, 0, 2, 2])
add_trailing_padding(y[77])
# 78
y.append([2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2, 2])
add_trailing_padding(y[78])
# 79
y.append(
    [2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 1, 1, 2, 2, 0, 2, 2,
     2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[79])
# 80
y.append(
    [2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     0, 2, 2, 2, 0, 1, 1, 1, 2, 0, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[80])
# 81
y.append([2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[81])
# 82
y.append([2, 0, 1, 2, 0, 1, 1, 1, 2, 2, 2, 2, 0, 2, 0, 2, 0, 1, 2, 0, 1, 2, 2])
add_trailing_padding(y[82])
# 83
y.append([2, 0, 1, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[83])
# 84
y.append([2, 2, 0, 1, 2, 2, 2, 0, 2, 0, 2, 2])
add_trailing_padding(y[84])
# 85
y.append([2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[85])
# 86
y.append([2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[86])
# 87
y.append([2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[87])
# 88
y.append([2, 0, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 2, 2, 2])
add_trailing_padding(y[88])
# 89
y.append(
    [2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 0,
     1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[89])
# 90
y.append(
    [2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1,
     2, 2, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[90])
# 91
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[91])
# 92
y.append([2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[92])
# 93
y.append([2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[93])
# 94
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 1, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[94])
# 95
y.append([2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[95])
# 96
y.append([2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 1, 0, 1, 2, 2, 2])
add_trailing_padding(y[96])
# 97
y.append([2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2])
add_trailing_padding(y[97])
# 98
y.append([2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[98])
# 99
y.append([2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[99])
# 100
y.append([2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[100])
# 101
y.append(
    [2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 1,
     1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 2, 2])
add_trailing_padding(y[101])
# 102
y.append([2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[102])
# 103
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1,
     2, 2, 2, 2, 2])
add_trailing_padding(y[103])
# 104
y.append(
    [2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 2, 0, 2, 2])
add_trailing_padding(y[104])
# 105
y.append([2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2])
add_trailing_padding(y[105])
# 106
y.append([2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[106])
# 107
y.append(
    [2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1,
     2, 2])
add_trailing_padding(y[107])
# 108
y.append([2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[108])
# 109
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[109])
# 110
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2])
add_trailing_padding(y[110])
# 111
y.append([2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2])
add_trailing_padding(y[111])
# 112
y.append(
    [2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2])
add_trailing_padding(y[112])
# 113
y.append(
    [2, 0, 1, 1, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 1, 2,
     2, 2, 2])
add_trailing_padding(y[113])
# 114
y.append([2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[114])
# 115
y.append([2, 0, 1, 1, 2, 0, 1, 2, 0, 2, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[115])
# 116
y.append([2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[116])
# 117
y.append([2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[117])
# 118
y.append([2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[118])
# 119
y.append([2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2])
add_trailing_padding(y[119])
# 120
y.append([2, 2, 0, 1, 2, 0, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[120])
# 121
y.append([2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2])
add_trailing_padding(y[121])
# 122
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[122])
# 123
y.append(
    [2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0,
     2, 2, 0, 2, 0, 1, 2, 2])
add_trailing_padding(y[123])
# 124
y.append([2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2])
add_trailing_padding(y[124])
# 125
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0,
     1, 2, 2])
add_trailing_padding(y[125])
# 126
y.append([2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[126])
# 127
y.append([2, 0, 1, 0, 1, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[127])
# 128
y.append(
    [2, 0, 1, 0, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2,
     2, 0, 1, 1, 1, 2, 2, 2])
add_trailing_padding(y[128])
# 129
y.append([2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[129])
# 130
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[130])
# 131
y.append([2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[131])
# 132
y.append([2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[132])
# 133
y.append([2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[133])
# 134
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[134])
# 135
y.append(
    [2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2,
     2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[135])
# 136
y.append([2, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[136])
# 137
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[137])
# 138
y.append([2, 2, 0, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[138])
# 139
y.append([2, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[139])
# 140
y.append([2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[140])
# 141
y.append([2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 1, 1, 2, 0, 2, 2, 0, 1, 2, 0, 2, 2, 2, 0, 2, 0, 2, 2, 2])
add_trailing_padding(y[141])
# 142
y.append([2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[142])
# 143
y.append([2, 0, 1, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[143])
# 144
y.append([2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[144])
# 145
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[145])
# 146
y.append(
    [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
     1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2])
add_trailing_padding(y[146])
# 147
y.append([2, 2, 0, 2, 2, 2, 0, 2, 0, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2])
add_trailing_padding(y[147])
# 148
y.append([2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2])
add_trailing_padding(y[148])
# 149
y.append([2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[149])
# 150
y.append([2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[150])
# 151
y.append([2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[151])
# 152
y.append([2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[152])
# 153
y.append([2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[153])
# 154
y.append([2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2, 0, 1, 2, 0, 2, 0, 2, 2, 0, 1, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[154])
# 155
y.append([2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[155])
# 156
y.append([2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[156])
# 157
y.append([2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2])
add_trailing_padding(y[157])
# 158
y.append([2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[158])
# 159
y.append([2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[159])
# 160
y.append([2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[160])
# 161
y.append([2, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[161])
# 162
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2,
     2, 2, 0, 2, 2])
add_trailing_padding(y[162])
# 163
y.append([2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[163])
# 164
y.append([2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[164])
# 165
y.append([2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[165])
# 166
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     0, 1, 2, 2])
add_trailing_padding(y[166])
# 167
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2])
add_trailing_padding(y[167])
# 168
y.append([2, 0, 1, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[168])
# 169
y.append([2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 1, 1, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[169])
# 170
y.append(
    [2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[170])
# 171
y.append([2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2])
add_trailing_padding(y[171])
# 172
y.append([2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2])
add_trailing_padding(y[172])
# 173
y.append([2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[173])
# 174
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[174])
# 175
y.append([2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[175])
# 176
y.append([2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[176])
# 177
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[177])
# 178
y.append([2, 0, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[178])
# 179
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[179])
# 180
y.append([2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2])
add_trailing_padding(y[180])
# 181
y.append([2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2])
add_trailing_padding(y[181])
# 182
y.append([2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[182])
# 183
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2])
add_trailing_padding(y[183])
# 184
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2])
add_trailing_padding(y[184])
# 185
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2])
add_trailing_padding(y[185])
# 186
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[186])
# 187
y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[187])
# 188
y.append([2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2])
add_trailing_padding(y[188])
# 189
y.append([2, 2, 0, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[189])
# 190
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[190])
# 191
y.append([2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[191])
# 192
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[192])
# 193
y.append([2, 0, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[193])
# 194
y.append([2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[194])
# 195
y.append(
    [2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[195])
# 196
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[196])
# 197
y.append([2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[197])
# 198
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2,
     2, 2, 2, 2])
add_trailing_padding(y[198])
# 199
y.append([2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[199])
# 200
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2,
     0, 1, 1, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[200])
# 201
y.append(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1,
     1, 2, 2, 2])
add_trailing_padding(y[201])
# 202
y.append([2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[202])
# 203
y.append([2, 0, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[203])
# 204
y.append([2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[204])
# 205
y.append([2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[205])
# 206
y.append([2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 0, 1, 2, 2])
add_trailing_padding(y[206])
# 207
y.append([2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[207])
# 208
y.append([2, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[208])
# 209
y.append([2, 0, 2, 0, 1, 2, 0, 1, 1, 1, 2, 2, 0, 2, 0, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[209])
# 210
y.append([2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[210])
# 211
y.append(
    [2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2,
     0, 1, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[211])
# 212
y.append([2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2])
add_trailing_padding(y[212])
# 213
y.append([2, 2, 2, 2, 0, 2, 0, 2, 2])
add_trailing_padding(y[213])
# 214
y.append([2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[214])
# 215
y.append([2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[215])
# 216
y.append([2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2])
add_trailing_padding(y[216])
# 217
y.append([2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[217])
# 218
y.append([2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[218])
# 219
y.append([2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[219])
# 220
y.append([2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[220])
# 221
y.append([2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[221])
# 222
y.append([2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[222])
# 223
y.append(
    [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2,
     2, 2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2])
add_trailing_padding(y[223])
# 224
y.append([2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[224])
# 225
y.append(
    [2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2])
add_trailing_padding(y[225])
# 226
y.append([2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2, 0, 2, 2])
add_trailing_padding(y[226])
# 227
y.append(
    [2, 2, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2])
add_trailing_padding(y[227])
# 228
y.append([2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2, 2])
add_trailing_padding(y[228])
# 229
y.append(
    [2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[229])
# 230
y.append(
    [2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 2, 2,
     2, 0, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[230])
# 231
y.append(
    [2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1,
     1, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[231])
# 232
y.append([2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[232])
# 233
y.append(
    [2, 2, 2, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2,
     2, 2, 2, 0, 1, 2, 2, 0, 2, 2])
add_trailing_padding(y[233])
# 234
y.append([2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2])
add_trailing_padding(y[234])
# 235
y.append([2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2])
add_trailing_padding(y[235])
# 236
y.append([2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[236])
# 237
y.append(
    [2, 2, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 1,
     1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[237])
# 238
y.append(
    [2, 2, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
add_trailing_padding(y[238])
# 239
y.append([])

# Fine Tune BERT
checkpoint = "bert-base-uncased"
bert = TFAutoModel.from_pretrained(checkpoint)
inputs = tf.keras.layers.Input(shape=(max_length,), name="input_ids", dtype="int32")
mask = tf.keras.layers.Input(shape=(max_length,), name="attention_mask", dtype="int32")
embeddings = bert(input_ids=inputs, attention_mask=mask)[0]
X = tf.keras.layers.Dense(128, activation="relu")(embeddings)
X = tf.keras.layers.Dropout(0.2)(X)
outputs = tf.keras.layers.Dense(4, activation="softmax", name="output")(X)
model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs)
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", precision, recall])
model.summary()
# #
# Just trying out
x = np.array(input_ids[0:len(y)])
x_mask = np.array(attention_mask[0:len(y)])

x = np.concatenate((x, x_mask), axis=1)

y_distribution = []
for labels in y:
    y_distribution.extend(labels)
print(Counter(y_distribution))

# One hot the labels so that the model can calculate loss and accuracy
y_one_hot = []
for i in range(len(y)):
    y_one_hot.append([])
    for j in range(len(y[i])):
        one_hot = [0] * len(LABELS)
        one_hot[y[i][j]] = 1
        y_one_hot[i].append(one_hot)

y = np.array(y)
y_one_hot = np.array(y_one_hot)

# Split it into training and validation set
x_train, x_val, y_train, y_val = train_test_split(x, y_one_hot, test_size=0.2, shuffle=True)

x_train_mask = x_train[:, max_length:]
x_train = x_train[:, 0:max_length]
x_val_mask = x_val[:, max_length:]
x_val = x_val[:, 0:max_length]
#
model.fit([x_train, x_train_mask], y_train, validation_data=([x_val, x_val_mask], y_val), batch_size=8, epochs=10)
# # #
# test_sentence = 'web apis'
# tokenized_test_sentence = tokenizer(test_sentence, padding="max_length", max_length=max_length)
# test_sentence_ids = np.array([tokenized_test_sentence["input_ids"]])
# test_sentence_mask = np.array([tokenized_test_sentence["attention_mask"]])
# test_sentence_tokens = []
# for ids in test_sentence_ids:
#     test_sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))
# print(np.argmax(model.predict(x=[test_sentence_ids, test_sentence_mask]), axis=-1))
# print(test_sentence_tokens)
np.save("data/sentence_tokens", sentence_tokens[:len(y)])
np.save("data/entity_labels", y)
