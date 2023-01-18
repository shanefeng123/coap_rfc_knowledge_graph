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

class MeditationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])
def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         labels=labels)
    test_loss = test_outputs.loss
    predictions = torch.argmax(test_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / labels.shape[0]
    return test_loss, predictions, accuracy, labels

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

y = json.load(open('../data/mqtt_relation.json', 'r'))
processed_sentences = []
labels = []
for i in range(len(y)):
    # if "sentiment" not in y[i].keys():
    #     print(y[i]["id"])
    processed_sentences.append(y[i]["text"])
    labels.append(RELATIONS[y[i]["sentiment"]])

y = labels
#
print(len(y))
print(f"no relation class percentage: {Counter(labels)[0] / len(labels)}")

for i in range(len(processed_sentences)):
    processed_sentences[i] = processed_sentences[i].replace("[CLS]", "")
    processed_sentences[i] = processed_sentences[i].replace("[SEP]", "")
    processed_sentences[i] = processed_sentences[i].replace("[PAD]", "")
    processed_sentences[i] = re.sub(' +', ' ', processed_sentences[i])
    processed_sentences[i] = processed_sentences[i].strip()

for i in range(len(processed_sentences)):
    processed_sentences[i] = tokenizer.convert_tokens_to_string(processed_sentences[i])

inputs = tokenizer(processed_sentences, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
#
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

labels = torch.LongTensor(labels)
inputs["labels"] = labels

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cpu":
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")
model = torch.load("../model/relation_extractor.pt", map_location=device)

dataset = MeditationDataset(inputs)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

test_loop = tqdm(test_loader, leave=True)
overall_test_loss = 0
epoch_test_predictions = None
epoch_test_labels = None
num_of_test_batches = len(test_loader)
for test_batch in test_loop:
    model.eval()
    test_loss, test_predictions, test_accuracy, test_labels = test(test_batch, model)
    test_loop.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())
    overall_test_loss += test_loss.item()

    if epoch_test_predictions is None:
        epoch_test_predictions = test_predictions
        epoch_test_labels = test_labels
    else:
        epoch_test_predictions = torch.cat((epoch_test_predictions, test_predictions), dim=0)
        epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

average_test_loss = overall_test_loss / num_of_test_batches
epoch_test_accuracy = torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels)) / epoch_test_labels.shape[0]

print(f"average test loss: {average_test_loss}")
print(f"test accuracy: {epoch_test_accuracy.item()}")
print("Testing report")
print(metrics.classification_report(epoch_test_labels.tolist(), epoch_test_predictions.tolist(), zero_division=0))