import pdfplumber
import nltk
import re
import torch
from transformers import BertTokenizer

LABELS = ["B-entity", "I-entity", "Other", "PAD"]
all_entity_indexes = []


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


mqtt_spec = []

with pdfplumber.open("data/mqtt_specification.pdf") as pdf:
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
mqtt_spec_sentences = mqtt_spec_sentences[:50]

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = torch.load("model/entity_extractor.pt")

inputs = tokenizer(mqtt_spec_sentences, padding="max_length", truncation=True, return_tensors="pt")

input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y = []

y.append(annotate_entity(sentence_tokens[0], []))
y.append(annotate_entity(sentence_tokens[1], []))
y.append(annotate_entity(sentence_tokens[2], []))
y.append(annotate_entity(sentence_tokens[3], [(1, 2), (10, 11), (17, 19)]))
y.append(annotate_entity(sentence_tokens[4], [(5, 6), (9, 9)]))
y.append(annotate_entity(sentence_tokens[5], []))
y.append(annotate_entity(sentence_tokens[6], [(8, 9)]))
y.append(annotate_entity(sentence_tokens[7], [(1, 2), (9, 12)]))
y.append(annotate_entity(sentence_tokens[8], [(3, 4), (8, 10), (13, 13), (17, 19), (21, 23), (29, 29), (33, 35)]))
y.append(annotate_entity(sentence_tokens[9], [(1, 2), (10, 12)]))
y.append(annotate_entity(sentence_tokens[10], [(2, 3), (8, 9), (12, 12), (15, 17), (20, 22)]))
y.append(annotate_entity(sentence_tokens[11], [(8, 9)]))
y.append(annotate_entity(sentence_tokens[12], [(12, 14)]))
y.append(annotate_entity(sentence_tokens[13], [(4, 5), (8, 8)]))
y.append(annotate_entity(sentence_tokens[14], [(1, 1), (11, 13), (15, 17), (20, 22), (24, 26), (30, 32)]))
y.append(annotate_entity(sentence_tokens[15], [(2, 2), (6, 8), (10, 12)]))
y.append(annotate_entity(sentence_tokens[16], [(3, 5), (8, 10)]))
y.append(annotate_entity(sentence_tokens[17], [(3, 5), (7, 10), (13, 15)]))
y.append(annotate_entity(sentence_tokens[18], [(3, 5), (8, 12)]))
y.append(annotate_entity(sentence_tokens[19], [(4, 5), (8, 9)]))
y.append(annotate_entity(sentence_tokens[20], [(1, 1), (9, 10), (13, 13)]))
y.append(annotate_entity(sentence_tokens[21], [(2, 2), (9, 10), (17, 19), (22, 23), (26, 26)]))
y.append(annotate_entity(sentence_tokens[22], [(1, 2), (5, 6), (9, 12), (16, 18)]))
y.append(annotate_entity(sentence_tokens[23], [(2, 3), (9, 9)]))
y.append(annotate_entity(sentence_tokens[24], [(2, 2), (8, 9)]))
y.append(annotate_entity(sentence_tokens[25], [(2, 3), (6, 6), (10, 13)]))
y.append(annotate_entity(sentence_tokens[26], [(1, 5), (8, 12), (15, 18), (22, 24)]))
y.append(annotate_entity(sentence_tokens[27], [(2, 6), (14, 14), (21, 23)]))
y.append(annotate_entity(sentence_tokens[28], [(2, 3), (7, 11), (17, 18), (24, 24)]))
y.append(annotate_entity(sentence_tokens[29], [(2, 2), (11, 15), (20, 25), (27, 29)]))
y.append(annotate_entity(sentence_tokens[30], [(1, 4), (7, 10), (13, 14), (17, 20)]))
y.append(annotate_entity(sentence_tokens[31], [(4, 4), (10, 12)]))
y.append(annotate_entity(sentence_tokens[32], [(17, 20)]))
y.append(annotate_entity(sentence_tokens[33], [(1, 3), (10, 11), (17, 19), (23, 23)]))
y.append(annotate_entity(sentence_tokens[34], [(1, 4), (11, 12)]))
y.append(annotate_entity(sentence_tokens[35], [(2, 5)]))
y.append(annotate_entity(sentence_tokens[36], [(1, 6), (9, 9), (17, 18)]))
y.append(annotate_entity(sentence_tokens[37], [(2, 4), (11, 16), (21, 26), (31, 33)]))
y.append(annotate_entity(sentence_tokens[38], [(1, 4), (7, 8)]))
y.append(annotate_entity(sentence_tokens[39], [(11, 12)]))
y.append(annotate_entity(sentence_tokens[40], [(1, 4), (7, 7), (13, 13), (29, 29), (38, 39), (41, 41)]))
y.append(annotate_entity(sentence_tokens[41], [(11, 12)]))
y.append(annotate_entity(sentence_tokens[42], [(1, 2), (5, 6), (12, 12), (15, 16), (23, 24)]))
y.append(annotate_entity(sentence_tokens[43], [(15, 17)]))
y.append(annotate_entity(sentence_tokens[44], [(1, 9), (14, 19), (21, 27), (35, 42)]))
y.append(annotate_entity(sentence_tokens[45], [(15, 23)]))
y.append(annotate_entity(sentence_tokens[46], [(1, 1)]))
y.append(annotate_entity(sentence_tokens[47], [(6, 11), (12, 17)]))
y.append(annotate_entity(sentence_tokens[48], [(15, 19), (21, 22), (27, 32), (34, 35)]))
y.append(annotate_entity(sentence_tokens[49], [(6, 11), (12, 17)]))

labels = torch.LongTensor(y)
inputs["labels"] = labels

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cpu":
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")
model.to(device)
model.eval()
test_loss, predictions, accuracy, labels = test(inputs, model)
print(accuracy.item())