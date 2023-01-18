import random
import pdfplumber
import nltk
import re
import torch
from transformers import BertTokenizer
import pickle
from sklearn import metrics

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
LABELS = ["B-entity", "I-entity", "Other", "PAD"]
all_entity_indexes = []


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         labels=labels)
    test_loss = test_outputs.loss
    predictions = torch.argmax(test_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / (
            labels.shape[0] * labels.shape[1])
    return test_loss, predictions, accuracy, labels


def annotate_entity(sentence, entity_indexes):
    annotations = [None] * len(sentence)
    # Annotate all the pad to class 3
    pad_start = sentence.index("[PAD]")
    for i in range(pad_start, len(annotations)):
        annotations[i] = 3

    for entity_index in entity_indexes:
        start = entity_index[0]
        end = entity_index[1]
        annotations[start] = 0
        for i in range(start + 1, end + 1):
            annotations[i] = 1

    for i in range(len(annotations)):
        if annotations[i] is None:
            annotations[i] = 2

    all_entity_indexes.append(entity_indexes)
    return annotations


mqtt_spec = []

with pdfplumber.open("../data/mqtt_specification.pdf") as pdf:
    pages = pdf.pages[10: 118]
    for page in pages:
        text = page.extract_text(layout=False)
        text = text.split("\n")
        for line in text:
            line = line.strip()

            alpha = any(c.isalpha() for c in line)
            if not alpha:
                line = ""

            if line.startswith("mqtt-v5"):
                line = ""

            if line.startswith("Standards Track Work Product"):
                line = ""

            if line == "":
                continue

            separate = line.split(" ", 1)
            if separate[0].isdigit():
                mqtt_spec.append(separate[1])
            else:
                mqtt_spec.append(line)

mqtt_spec = "\n".join(mqtt_spec)
mqtt_spec_sentences = nltk.sent_tokenize(mqtt_spec, "english")

for i in range(len(mqtt_spec_sentences)):
    mqtt_spec_sentences[i] = mqtt_spec_sentences[i].strip()
    mqtt_spec_sentences[i] = mqtt_spec_sentences[i].replace("\n", " ")
    mqtt_spec_sentences[i] = re.sub(' +', ' ', mqtt_spec_sentences[i])

    alpha = any(c.isalpha() for c in mqtt_spec_sentences[i])
    if not alpha:
        mqtt_spec_sentences[i] = ""

    if "Figure" in mqtt_spec_sentences[i]:
        mqtt_spec_sentences[i] = ""

mqtt_spec_sentences = [sentence for sentence in mqtt_spec_sentences if sentence != ""]
mqtt_spec_sentences = mqtt_spec_sentences[:46] + mqtt_spec_sentences[49:]

mqtt_rule_sentences = []
for sentence in mqtt_spec_sentences:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            mqtt_rule_sentences.append(sentence)
            break

# Start from here, sample 20% mqtt rule sentences
random.seed(4)
sampled_mqtt_rule_sentences = random.sample(mqtt_rule_sentences, 63)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(sampled_mqtt_rule_sentences, padding="max_length", truncation=True, return_tensors="pt")

input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y = []
y.append(annotate_entity(sentence_tokens[0], [(3, 4), (15, 15), (18, 19), (33, 33), (41, 42)]))
y.append(annotate_entity(sentence_tokens[1], [(3, 4), (7, 11), (20, 20), (26, 31), (43, 44)]))  #
y.append(
    annotate_entity(sentence_tokens[2], [(3, 4), (7, 11), (11, 17), (20, 21), (29, 29), (32, 33), (35, 41), (47, 52)]))
y.append(annotate_entity(sentence_tokens[3], [(3, 3), (5, 10), (25, 29), (35, 36)]))
y.append(annotate_entity(sentence_tokens[4], [(2, 6), (13, 17), (21, 26)]))
y.append(annotate_entity(sentence_tokens[5], [(3, 3), (6, 9), (23, 24), (27, 31), (33, 35)]))
y.append(annotate_entity(sentence_tokens[6], [(2, 2), (10, 13), (16, 21), (24, 25), (28, 32)]))
y.append(annotate_entity(sentence_tokens[7], [(3, 4), (10, 12), (19, 19), (25, 26), (29, 30)]))
y.append(annotate_entity(sentence_tokens[8], [(3, 4)]))
y.append(annotate_entity(sentence_tokens[9], [(2, 2), (9, 15), (19, 24)]))
y.append(annotate_entity(sentence_tokens[10],
                         [(4, 4), (12, 16), (30, 31), (35, 35), (39, 40), (43, 44), (47, 47), (53, 54), (68, 69)]))
y.append(annotate_entity(sentence_tokens[11], [(2, 2), (8, 10), (15, 20), (25, 26), (29, 30)]))
y.append(annotate_entity(sentence_tokens[12], [(13, 13), (20, 21)]))
y.append(annotate_entity(sentence_tokens[13], [(3, 3), (8, 10), (13, 13), (19, 21), (25, 26), (28, 31)]))
y.append(annotate_entity(sentence_tokens[14], [(13, 13), (20, 21)]))
y.append(annotate_entity(sentence_tokens[15], [(15, 18)]))
y.append(annotate_entity(sentence_tokens[16], [(7, 9), (23, 25), (29, 31)]))
y.append(annotate_entity(sentence_tokens[17], [(2, 3), (9, 13), (24, 29), (35, 39)]))
y.append(annotate_entity(sentence_tokens[18], [(4, 5), (8, 9), (12, 13), (24, 25)]))
y.append(annotate_entity(sentence_tokens[19], [(2, 4), (11, 13), (16, 17), (27, 30), (33, 34), (37, 37)]))
y.append(annotate_entity(sentence_tokens[20], [(2, 2), (8, 14), (19, 19), (22, 23)]))
y.append(annotate_entity(sentence_tokens[21], [(2, 2), (19, 24), (27, 31)]))
y.append(annotate_entity(sentence_tokens[22], [(4, 4), (8, 12), (15, 17), (21, 26), (36, 40)]))
y.append(annotate_entity(sentence_tokens[23], [(2, 5), (9, 10), (18, 21)]))
y.append(annotate_entity(sentence_tokens[24], [(11, 16), (19, 22)]))
y.append(annotate_entity(sentence_tokens[25], [(7, 11), (13, 14)]))
y.append(
    annotate_entity(sentence_tokens[26], [(3, 3), (6, 6), (8, 9), (15, 15), (20, 21), (26, 31), (43, 44), (47, 52)]))
y.append(annotate_entity(sentence_tokens[27], [(2, 7), (11, 12), (15, 15), (23, 27)]))
y.append(annotate_entity(sentence_tokens[28], [(2, 7), (11, 11), (14, 15), (21, 22), (25, 28), (30, 33)]))
y.append(annotate_entity(sentence_tokens[29], [(2, 2), (9, 9), (11, 15), (18, 19)]))
y.append(annotate_entity(sentence_tokens[30], [(10, 10), (18, 24), (27, 28), (32, 33)]))
y.append(annotate_entity(sentence_tokens[31], [(2, 2), (18, 19)]))
y.append(annotate_entity(sentence_tokens[32], [(2, 2), (18, 19)]))
y.append(annotate_entity(sentence_tokens[33], [(2, 3), (9, 11), (18, 21), (31, 32)]))
y.append(annotate_entity(sentence_tokens[34], [(6, 11), (18, 18), (24, 28), (35, 36), (41, 45), (51, 56)]))
y.append(annotate_entity(sentence_tokens[35], [(3, 3), (6, 11), (14, 15), (27, 28)]))
y.append(annotate_entity(sentence_tokens[36], [(3, 3), (6, 11), (14, 18), (24, 26), (36, 40), (44, 45)]))
y.append(annotate_entity(sentence_tokens[37], [(7, 12), (15, 19), (21, 23), (26, 28)]))
y.append(annotate_entity(sentence_tokens[38], [(2, 2), (8, 12), (23, 28), (34, 39)]))
y.append(annotate_entity(sentence_tokens[39], [(3, 4), (12, 12), (32, 33)]))
y.append(annotate_entity(sentence_tokens[40], [(3, 7), (12, 13), (20, 20), (28, 29)]))
y.append(annotate_entity(sentence_tokens[41], [(5, 5), (11, 11), (14, 15), (20, 22), (27, 28)]))
y.append(annotate_entity(sentence_tokens[42], [(2, 5), (20, 21)]))
y.append(annotate_entity(sentence_tokens[43], [(2, 2), (10, 12), (17, 18)]))
y.append(annotate_entity(sentence_tokens[44], [(3, 7), (11, 12), (18, 19), (25, 26), (33, 38), (41, 45)]))
y.append(annotate_entity(sentence_tokens[45],
                         [(2, 3), (12, 14), (15, 18), (20, 23), (24, 29), (36, 40), (42, 46), (49, 52), (55, 56),
                          (63, 63)]))
y.append(annotate_entity(sentence_tokens[46], [(4, 5), (10, 15), (32, 37), (45, 47), (49, 53)]))

y.append(annotate_entity(sentence_tokens[47], [(4, 5), (9, 14), (22, 26), (29, 35)]))
y.append(annotate_entity(sentence_tokens[48], [(2, 3), (5, 5), (8, 14), (22, 27), (28, 39)]))
y.append(annotate_entity(sentence_tokens[49], [(6, 6), (9, 11), (17, 23), (26, 27)]))
y.append(annotate_entity(sentence_tokens[50], [(2, 2), (8, 11), (16, 16), (19, 20)]))
y.append(annotate_entity(sentence_tokens[51], [(20, 26), (29, 33)]))
y.append(annotate_entity(sentence_tokens[52], [(8, 9), (11, 12), (14, 14), (23, 24), (27, 28)]))
y.append(annotate_entity(sentence_tokens[53], [(3, 4), (9, 14), (17, 17), (26, 27), (34, 35)]))
y.append(annotate_entity(sentence_tokens[54], [(3, 7), (14, 19), (23, 24), (27, 27), (30, 30), (45, 46)]))
y.append(annotate_entity(sentence_tokens[55], [(3, 3), (10, 10), (18, 23), (25, 30), (34, 35), (39, 44)]))
y.append(annotate_entity(sentence_tokens[56], [(2, 4), (7, 12), (16, 16), (22, 23), (29, 30), (33, 36)]))
y.append(annotate_entity(sentence_tokens[57], [(3, 7), (11, 12), (19, 20), (22, 22), (30, 30), (35, 35)]))
y.append(annotate_entity(sentence_tokens[58], [(3, 4), (12, 12), (17, 22), (25, 26), (34, 35), (45, 46), (49, 50)]))
y.append(annotate_entity(sentence_tokens[59], [(3, 7), (11, 12), (20, 20), (24, 28), (31, 31), (39, 40), (47, 47)]))
y.append(annotate_entity(sentence_tokens[60], [(3, 8), (20, 24), (31, 32)]))
y.append(annotate_entity(sentence_tokens[61], [(2, 7), (20, 24), (14, 14), (23, 24), (27, 28)]))
y.append(annotate_entity(sentence_tokens[62], [(2, 3), (14, 14), (17, 17), (28, 28), (34, 34), (37, 38), (41, 43)]))

input_ids = inputs["input_ids"]

input_ids_list = input_ids.tolist()

with open("../data/mqtt_sentence_input_ids_list", "wb") as file:
    pickle.dump(input_ids_list, file)

with open("../data/mqtt_sentence_entity_indexes", "wb") as file:
    pickle.dump(all_entity_indexes, file)
#
# #
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.load("../model/entity_extractor.pt", map_location=device)
#
labels = torch.LongTensor(y)
inputs["labels"] = labels
#
model.to(device)
model.eval()
test_loss, predictions, accuracy, labels = test(inputs, model)

test_precision = metrics.precision_score(torch.flatten(labels).tolist(),
                                         torch.flatten(predictions).tolist(), average="macro")

test_recall = metrics.recall_score(torch.flatten(labels).tolist(),
                                   torch.flatten(predictions).tolist(), average="macro")

test_f1 = metrics.f1_score(torch.flatten(labels).tolist(),
                           torch.flatten(predictions).tolist(), average="macro")

print(
    f"Test loss: {test_loss}, Test accuracy: {accuracy}, Test precision: {test_precision}, Test recall: {test_recall}, Test f1: {test_f1}")

print(
    metrics.classification_report(torch.flatten(labels).tolist(), torch.flatten(predictions).tolist(), zero_division=0))

with open(r"../results/MQTT_entity_benchmark.txt", "a") as file:
    file.write(
        f"Test loss: {test_loss}, Test accuracy: {accuracy}, Test precision: {test_precision}, Test recall: {test_recall}, Test f1: {test_f1}")
    file.write("\n")

# test_sentence = "Both Block1 and Block2 Options can be present in both the request and response messages."
# test_inputs = tokenizer(test_sentence, return_tensors="pt")
# test_input_ids = test_inputs["input_ids"].to(device)
# test_token_type_ids = test_inputs["token_type_ids"].to(device)
# test_attention_mask = test_inputs["attention_mask"].to(device)
# test_outputs = model(input_ids=test_input_ids, token_type_ids=test_token_type_ids, attention_mask=test_attention_mask)
# test_predictions = torch.argmax(test_outputs.logits, dim=-1)
# test_sentence_tokens = []
# for ids in test_input_ids:
#     test_sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))
# print(test_sentence_tokens)
# print(test_predictions)
