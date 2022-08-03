import numpy as np
from itertools import combinations
from transformers import BertForPreTraining, BertTokenizer
import torch
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


# Try out the similarity matching approach
print(Counter(relation_labels).most_common())
# Only take the relations that have more than 8 samples
min_num_samples = 8
common_relations = []
for key, value in Counter(relation_labels).items():
    if value >= min_num_samples:
        common_relations.append(key)
common_relations.sort()
num_of_common_relations = len(common_relations)


# Take out the common relation sequences and only sample 30 sequences for the "no relation" class
no_relation_sequences = []
X = []
y = []
for i in range(len(relation_labels)):
    if relation_labels[i] in common_relations:
        if relation_labels[i] == 0:
            no_relation_sequences.append(squeeze_processed_sentences[i])
        else:
            X.append(squeeze_processed_sentences[i])
            y.append(common_relations.index(relation_labels[i]))

no_relation_sequences = random.choices(no_relation_sequences, k=30)
X.extend(no_relation_sequences)
y.extend([0] * 30)

# Load the model and tokenizer. Add the new special tokens. Construct the attention mask
max_length = len(X[0])
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"], special_tokens=True)
bert = torch.load("coap_BERT.pt", map_location=torch.device('cpu')).bert
# print(bert.embeddings.vocab_size)
bert.resize_token_embeddings(len(tokenizer))
print(bert.embeddings.word_embeddings)
attention_masks = []
for sentence in X:
    pad_index = sentence.index("[PAD]")
    mask = [1] * pad_index
    mask.extend([0] * (max_length - pad_index))
    attention_masks.append(mask)

# Transform the sequence of tokens into token numbers
input_ids = []
for sentence in X:
    ids = []
    for word in sentence:
        if word in ["[E1]", "[/E1]", "[E2]", "[/E2]"]:
            ids.append(tokenizer(word)["input_ids"][1])
        else:
            ids.append(tokenizer.vocab[word])
    input_ids.append(ids)

#
# inputs = {}
# X = torch.tensor(input_ids)
# attention_masks = torch.tensor(attention_masks)
# y = torch.tensor(y)
#
# inputs["input_ids"] = X
# inputs["label"] = y
# inputs["attention_mask"] = attention_masks
# outputs = bert(input_ids=X, attention_mask=attention_masks)
# inputs["vector_representation"] = outputs.pooler_output
# inputs["all_tokens_embeddings"] = outputs.last_hidden_state


# class MeditationDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#
#     def __len__(self):
#         return len(self.encodings["input_ids"])
#
#
# dataset = MeditationDataset(inputs)
# train_length = int(len(dataset) * 0.8)
# test_length = len(dataset) - train_length
# train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])
#
# relation_vector_representations = []
# for i in range(len(common_relations)):
#     relation_vector_representations.append([])

# This is on pooler_output only. Should try using the mean of all tokens
# for i in range(len(train_set)):
#     sample = train_set[i]
#     vector_representation = sample["vector_representation"]
#     label = sample["label"]
#     relation_vector_representations[label.item()].append(vector_representation)

# for i in range(len(train_set)):
#     sample = train_set[i]
#     all_tokens_embeddings = sample["all_tokens_embeddings"]
#     attention_mask = sample["attention_mask"].bool()
#     valid_tokens_embeddings = all_tokens_embeddings[attention_mask, :]
#     mean_valid_tokens_embedding = torch.mean(valid_tokens_embeddings, dim=0)
#     label = sample["label"]
#     relation_vector_representations[label.item()].append(mean_valid_tokens_embedding)


# centroids = []
# for i in range(len(relation_vector_representations)):
#     centroids.append(torch.mean(torch.stack(relation_vector_representations[i]), dim=0))
#
# centroids = torch.stack(centroids)
# cosine_sim = torch.nn.CosineSimilarity(dim=1)
# train_predictions = []
# train_labels = train_set[:]["label"]
# for i in range(len(train_set)):
#     vector_representation = train_set[i]["vector_representation"]
#     similarities = cosine_sim(centroids, vector_representation)
#     prediction = torch.argmax(similarities)
#     train_predictions.append(prediction.item())
#
# train_predictions = torch.tensor(train_predictions)
# train_accuracy = torch.sum(torch.eq(train_labels, train_predictions)) / train_predictions.shape[0]
# print(train_accuracy.item())

# test_predictions = []


# # x = np.concatenate((x, x_mask), axis=1)
#
# y = np.array(y)
# y_one_hot = []
# for i in range(len(y)):
#     one_hot = [0] * len(RELATIONS)
#     one_hot[y[i]] = 1
#     y_one_hot.append(one_hot)
# y_one_hot = np.array(y_one_hot)
#
# model.fit([x, attention_masks], y_one_hot, batch_size=32, epochs=10)
# print(np.argmax(model.predict(x=[x, attention_masks]), axis=-1))
# print(Counter(relation_labels))
