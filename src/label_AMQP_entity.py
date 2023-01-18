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


amqp_spec = []
with pdfplumber.open("../data/amqp_specification.pdf") as pdf:
    pages = pdf.pages[16:119]
    for page in pages:
        text = page.extract_text(layout=False, x_tolerance=1)
        text = text.split("\n")
        for line in text:
            line = line.strip()

            alpha = any(c.isalpha() for c in line)
            if not alpha:
                line = ""

            if line.startswith("amqp-core"):
                line = ""

            if line.startswith("PART"):
                line = ""

            if line.startswith("0x"):
                line = ""

            if line.startswith("<type"):
                line = ""

            if line.startswith("label="):
                line = ""

            if line.startswith("<encoding"):
                line = ""

            if line.startswith("<descriptor"):
                line = ""

            if line.startswith("Standards Track Work Product"):
                line = ""

            if line == "":
                continue

            separate = line.split(" ", 1)
            if separate[0].isdigit():
                amqp_spec.append(separate[1])
            else:
                amqp_spec.append(line)

amqp_spec = "\n".join(amqp_spec)
amqp_spec_sentences = nltk.sent_tokenize(amqp_spec, "english")
for i in range(len(amqp_spec_sentences)):
    amqp_spec_sentences[i] = amqp_spec_sentences[i].strip()
    amqp_spec_sentences[i] = amqp_spec_sentences[i].replace("\n", " ")
    amqp_spec_sentences[i] = re.sub(' +', ' ', amqp_spec_sentences[i])

    alpha = any(c.isalpha() for c in amqp_spec_sentences[i])
    if not alpha:
        amqp_spec_sentences[i] = ""

    if "Figure" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

    if amqp_spec_sentences[i].startswith("</type>"):
        amqp_spec_sentences[i] = ""

    if amqp_spec_sentences[i].startswith("<field"):
        amqp_spec_sentences[i] = ""

    if "-->" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

    if "--+" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

    if "||" in amqp_spec_sentences[i]:
        amqp_spec_sentences[i] = ""

amqp_spec_sentences = [sentence for sentence in amqp_spec_sentences if sentence != ""]
amqp_rule_sentences = []
for sentence in amqp_spec_sentences:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            amqp_rule_sentences.append(sentence)
            break

print(len(amqp_rule_sentences))

# TODO Remeber to sample 20% of the rule sentences before annotating, and set a random seed
random.seed(4)
sampled_AMQP_rule_sentences = random.sample(amqp_rule_sentences, 67)
print(len(sampled_AMQP_rule_sentences))
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(sampled_AMQP_rule_sentences, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y = []
y.append(annotate_entity(sentence_tokens[0], [(10, 12)]))
y.append(annotate_entity(sentence_tokens[1], [(3, 3)]))
y.append(annotate_entity(sentence_tokens[2], [(2, 6)]))
y.append(annotate_entity(sentence_tokens[3], [(8, 9)]))
y.append(annotate_entity(sentence_tokens[4], [(5, 7), (10, 12), (20, 20), (28, 30)]))
y.append(annotate_entity(sentence_tokens[5], [(9, 11), (23, 24), (32, 35), (38, 42), (44, 48), (50, 53)]))
y.append(annotate_entity(sentence_tokens[6], [(5, 6), (10, 10)]))
y.append(annotate_entity(sentence_tokens[7], [(34, 34), (42, 42)]))
y.append(annotate_entity(sentence_tokens[8], [(5, 6), (11, 11), (18, 19), (26, 26)]))
y.append(annotate_entity(sentence_tokens[9], [(20, 20), (23, 24), (36, 36), (40, 40)]))
y.append(annotate_entity(sentence_tokens[10], [(43, 45), (52, 55)]))
y.append(annotate_entity(sentence_tokens[11], [(5, 5), (17, 17), (25, 25), (28, 35)]))
y.append(annotate_entity(sentence_tokens[12], [(23, 24)]))
y.append(annotate_entity(sentence_tokens[13], [(7, 7), (13, 13), (16, 18)]))
y.append(annotate_entity(sentence_tokens[14], [(1, 3), (5, 6), (9, 9), (27, 28), (33, 35)]))
y.append(annotate_entity(sentence_tokens[15], [(6, 6), (13, 13), (17, 19), (23, 23)]))
y.append(annotate_entity(sentence_tokens[16], [(5, 5), (17, 18)]))
y.append(annotate_entity(sentence_tokens[17], [(15, 16)]))
y.append(annotate_entity(sentence_tokens[18], [(7, 9), (29, 31)]))
y.append(annotate_entity(sentence_tokens[19], [(1, 5), (7, 11), (21, 21), (30, 31), (34, 36)]))
y.append(annotate_entity(sentence_tokens[20], [(22, 22), (23, 23)]))
y.append(annotate_entity(sentence_tokens[21], [(12, 12)]))
y.append(annotate_entity(sentence_tokens[22], [(4, 4), (8, 8), (18, 18)]))
y.append(annotate_entity(sentence_tokens[23], [(33, 34), (36, 40)]))
y.append(annotate_entity(sentence_tokens[24], [(7, 10), (11, 12), (30, 30)]))
y.append(annotate_entity(sentence_tokens[25], [(2, 3), (16, 18)]))
y.append(annotate_entity(sentence_tokens[26], [(1, 3), (11, 13)]))
y.append(annotate_entity(sentence_tokens[27], [(1, 1), (3, 5), (22, 22), (29, 31)]))
y.append(annotate_entity(sentence_tokens[28], [(7, 7), (25, 26)]))
y.append(annotate_entity(sentence_tokens[29], [(14, 15), (19, 19)]))
y.append(annotate_entity(sentence_tokens[30], [(14, 15)]))
y.append(annotate_entity(sentence_tokens[31], [(2, 4), (31, 32), (41, 41)]))
y.append(annotate_entity(sentence_tokens[32], [(2, 2), (12, 13), (21, 21)]))
y.append(annotate_entity(sentence_tokens[33], [(12, 14), (23, 26)]))
y.append(annotate_entity(sentence_tokens[34], [(14, 15), (19, 19), (25, 26), (28, 29)]))
y.append(annotate_entity(sentence_tokens[35], [(3, 4), (6, 6)]))
y.append(annotate_entity(sentence_tokens[36], [(2, 2), (12, 12), (15, 16)]))  #
y.append(annotate_entity(sentence_tokens[37], [(2, 2), (14, 16)]))
y.append(annotate_entity(sentence_tokens[38], [(3, 4), (10, 10), (15, 15), (23, 23), (28, 28)]))
y.append(annotate_entity(sentence_tokens[39], [(13, 15)]))
y.append(annotate_entity(sentence_tokens[40], [(1, 2), (4, 4), (18, 19), (23, 26), (39, 39), (45, 45)]))
y.append(annotate_entity(sentence_tokens[41], [(12, 12), (16, 17), (23, 23)]))
y.append(annotate_entity(sentence_tokens[42], [(17, 19)]))
y.append(annotate_entity(sentence_tokens[43], [(3, 3)]))
y.append(annotate_entity(sentence_tokens[44], [(7, 7), (34, 34)]))
y.append(annotate_entity(sentence_tokens[45], [(3, 3)]))
y.append(annotate_entity(sentence_tokens[46], [(14, 14), (16, 16), (24, 29), (42, 42)]))
y.append(annotate_entity(sentence_tokens[47], [(8, 8), (9, 9), (17, 19)]))
y.append(
    annotate_entity(sentence_tokens[48], [(4, 4), (6, 6), (9, 9), (11, 11), (14, 14), (26, 27), (33, 34), (37, 37)]))
y.append(
    annotate_entity(sentence_tokens[49], [(3, 3), (8, 10), (16, 17), (21, 21), (24, 24), (31, 32), (61, 62), (66, 66)]))
y.append(annotate_entity(sentence_tokens[50], [(1, 3), (11, 13), (16, 18)]))
y.append(annotate_entity(sentence_tokens[51], [(6, 6), (15, 15), (19, 19), (21, 21), (23, 23)]))
y.append(annotate_entity(sentence_tokens[52], [(14, 15), (18, 18)]))
y.append(annotate_entity(sentence_tokens[53], [(25, 25), (30, 32)]))
y.append(annotate_entity(sentence_tokens[54], [(2, 2), (18, 18)]))
y.append(annotate_entity(sentence_tokens[55], [(10, 10), (14, 14)]))
y.append(annotate_entity(sentence_tokens[56], []))
y.append(annotate_entity(sentence_tokens[57], [(1, 1), (3, 3)]))
y.append(annotate_entity(sentence_tokens[58], [(7, 7), (14, 14), (24, 24), (27, 31)]))
y.append(annotate_entity(sentence_tokens[59], [(2, 2), (12, 12), (19, 20), (22, 22)]))
y.append(annotate_entity(sentence_tokens[60], [(6, 6), (9, 9), (21, 22)]))
y.append(annotate_entity(sentence_tokens[61], [(8, 8), (13, 13), (22, 22)]))
y.append(annotate_entity(sentence_tokens[62], [(4, 6), (17, 19), (26, 26), (37, 39)]))
y.append(annotate_entity(sentence_tokens[63], [(12, 13), (28, 29), (33, 35)]))
y.append(annotate_entity(sentence_tokens[64], [(1, 5)]))
y.append(annotate_entity(sentence_tokens[65], [(2, 2), (5, 5), (15, 15), (26, 26), (47, 47), (57, 57)]))
y.append(annotate_entity(sentence_tokens[66], [(3, 4)]))
labels = torch.LongTensor(y)
inputs["labels"] = labels
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
input_ids = inputs["input_ids"]

input_ids_list = input_ids.tolist()

with open("../data/amqp_sentence_input_ids_list", "wb") as file:
    pickle.dump(input_ids_list, file)

with open("../data/amqp_sentence_entity_indexes", "wb") as file:
    pickle.dump(all_entity_indexes, file)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cpu":
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")
model = torch.load("../model/entity_extractor.pt", map_location=device)

labels = torch.LongTensor(y)
inputs["labels"] = labels

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

with open(r"../results/AMQP_entity_benchmark.txt", "a") as file:
    file.write(
        f"Test loss: {test_loss}, Test accuracy: {accuracy}, Test precision: {test_precision}, Test recall: {test_recall}, Test f1: {test_f1}")
    file.write("\n")
