import pickle

import torch
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
from itertools import combinations
import json
import random
from tqdm import tqdm
from collections import Counter


class MeditationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


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


RELATIONS = {"NO RELATION": 0, "EQUIVALENT": 1, "HAS TYPE": 2, "HAS FIELD": 3, "TYPE OF": 4, "FIELD OF": 5,
             "HAS FEATURE": 6, "FEATURE OF": 7}

all_entity_indexes = pickle.load(open('../data/coap_sentence_entity_indexes', 'rb'))
input_ids = pickle.load(open('../data/coap_sentence_input_ids_list', 'rb'))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"], special_tokens=True)
model = BertForSequenceClassification.from_pretrained("../model/iot_bert", num_labels=len(RELATIONS))
model.resize_token_embeddings(len(tokenizer))

sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

processed_sentences = []
for i in range(len(sentence_tokens)):
    processed_sentences.extend(add_entity_markers(sentence_tokens[i], all_entity_indexes[i]))

with open(r"../data/coap_sentences_with_markers.txt", "w") as file:
    for sentence in processed_sentences:
        file.write("%s\n" % " ".join(sentence))

y = json.load(open('../data/coap_relation_labels.json', 'r'))
y.reverse()
for i in range(len(y)):
    y[i] = RELATIONS[y[i]["sentiment"]]

print(f"no relation class percentage: {Counter(y)[0] / len(y)}")

processed_sentences = processed_sentences[:len(y)]

no_relation_sentences = []
X = []
labels = []
for i in range(len(y)):
    if y[i] == 0:
        no_relation_sentences.append(processed_sentences[i])
    else:
        X.append(processed_sentences[i])
        labels.append(y[i])

no_relation_sampling = len(no_relation_sentences)
no_relation_sentences = random.choices(no_relation_sentences, k=no_relation_sampling)
X.extend(no_relation_sentences)
processed_sentences = X
labels.extend([0] * no_relation_sampling)

# Remove all the padding tokens in processed sentences
for i in range(len(processed_sentences)):
    processed_sentences[i] = [token for token in processed_sentences[i] if
                              token != "[PAD]" and token != "[CLS]" and token != "[SEP]"]

for i in range(len(processed_sentences)):
    processed_sentences[i] = tokenizer.convert_tokens_to_string(processed_sentences[i])

inputs = tokenizer(processed_sentences, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]

sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

labels = torch.LongTensor(labels)
inputs["labels"] = labels


def train(batch, model, optimizer):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                          labels=labels)
    train_loss = train_outputs.loss
    train_loss.backward()
    optimizer.step()
    predictions = torch.argmax(train_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / labels.shape[0]
    return train_loss, predictions, accuracy, labels


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


dataset = MeditationDataset(inputs)
dataset_length = len(dataset)
train_length = int(dataset_length * 0.8)
test_length = dataset_length - train_length
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(100):
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    epoch_train_predictions = None
    epoch_train_labels = None
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        model.train()
        train_loss, train_predictions, train_accuracy, train_labels = train(train_batch, model, optimizer)
        train_loop.set_postfix(train_loss=train_loss.item(), train_accuracy=train_accuracy.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Epoch {epoch} train")

        if epoch_train_predictions is None:
            epoch_train_predictions = train_predictions
            epoch_train_labels = train_labels
        else:
            epoch_train_predictions = torch.cat((epoch_train_predictions, train_predictions), dim=0)
            epoch_train_labels = torch.cat((epoch_train_labels, train_labels), dim=0)

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
        test_loop.set_description(f"Epoch {epoch} test")

        if epoch_test_predictions is None:
            epoch_test_predictions = test_predictions
            epoch_test_labels = test_labels
        else:
            epoch_test_predictions = torch.cat((epoch_test_predictions, test_predictions), dim=0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    epoch_train_accuracy = torch.sum(torch.eq(epoch_train_predictions, epoch_train_labels)) / epoch_train_labels.shape[0]

    average_test_loss = overall_test_loss / num_of_test_batches
    epoch_test_accuracy = torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels)) / epoch_test_labels.shape[0]

    print(f"average train loss: {average_train_loss}")
    print(f"epoch train accuracy: {epoch_train_accuracy.item()}")

    print(f"average test loss: {average_test_loss}")
    print(f"epoch test accuracy: {epoch_test_accuracy.item()}")

    with open(r"../results/relation_extractor_results.txt", "a") as file:
        file.write(
            f"Epoch {epoch} average_train_loss: {average_train_loss} train_accuracy: {epoch_train_accuracy.item()}")
        file.write("\n")
        file.write(
            f"Epoch {epoch} average_test_loss: {average_test_loss} test_accuracy: {epoch_test_accuracy.item()}")
        file.write("\n")

    if epoch_test_accuracy.item() > 0.8:
        break

torch.save(model, r"../model/relation_extractor.pt")