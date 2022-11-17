import pickle

import torch
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
from itertools import combinations
import re
import json
import random
from tqdm import tqdm
from collections import Counter
from sklearn import metrics

RELATIONS = {"NO RELATION": 0, "EQUIVALENT": 1, "HAS TYPE": 2, "HAS FIELD": 3, "TYPE OF": 4, "FIELD OF": 5,
             "HAS FEATURE": 6, "FEATURE OF": 7}

def add_entity_markers(sentence, entity_indexes):
    processed_sentences = []
    positions_combinations = combinations(entity_indexes, 2)
    for com in positions_combinations:
        first_entity_positions = com[0]
        second_entity_positions = com[1]
        first_entity_positions = (first_entity_positions[0], first_entity_positions[1] + 2)
        second_entity_positions = (second_entity_positions[0] + 2, second_entity_positions[1] + 4)
        processed_sentence = sentence.copy()
        processed_sentence.insert(first_entity_positions[0], "[E1]")
        processed_sentence.insert(first_entity_positions[1], "[/E1]")
        processed_sentence.insert(second_entity_positions[0], "[E2]")
        processed_sentence.insert(second_entity_positions[1], "[/E2]")
        processed_sentences.append(processed_sentence)
    return processed_sentences


all_entity_indexes = pickle.load(open('../data/mqtt_sentence_entity_indexes', 'rb'))
input_ids = pickle.load(open('../data/mqtt_sentence_input_ids_list', 'rb'))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"], special_tokens=True)

sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

processed_sentences = []
for i in range(len(sentence_tokens)):
    processed_sentences.extend(add_entity_markers(sentence_tokens[i], all_entity_indexes[i]))

for i in range(len(processed_sentences)):
    processed_sentences[i] = [token for token in processed_sentences[i] if
                              token != "[PAD]" and token != "[CLS]" and token != "[SEP]"]
    processed_sentences[i] = tokenizer.convert_tokens_to_ids(processed_sentences[i])
    processed_sentences[i] = tokenizer.decode(processed_sentences[i])
    processed_sentences[i] = re.sub(' +', ' ', processed_sentences[i])

with open(r"../data/mqtt_sentences_with_markers.txt", "w") as file:
    for sentence in processed_sentences:
        file.write(sentence)
        file.write("\n")


