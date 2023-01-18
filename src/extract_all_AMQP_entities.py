import random
import pdfplumber
import nltk
import re
import torch
from transformers import BertTokenizer
import pickle
from sklearn import metrics
from tqdm import tqdm

class MeditationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
LABELS = ["B-entity", "I-entity", "Other", "PAD"]
all_entity_indexes = []

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model/entity_extractor.pt", map_location=device)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(amqp_spec_sentences, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

dataset = MeditationDataset(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
loop = tqdm(dataloader, leave=True)
entity_predictions = []
for batch in loop:
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)
    entity_predictions.extend(predictions.tolist())

input_ids = inputs["input_ids"]

# Extract entities from the sentences with the predictions
entities = []
for i in range(len(input_ids)):
    sentence_token_ids = input_ids[i]
    sentence_entity_predictions = entity_predictions[i]
    current_entity_index = []
    current_entity = None
    for j in range(len(sentence_token_ids)):
        if sentence_entity_predictions[j] == 0 and current_entity_index == []:
            current_entity_index.append(j)
        elif sentence_entity_predictions[j] == 0 and current_entity_index != []:
            current_entity_index.append(j)
            current_entity_index = [current_entity_index[0], current_entity_index[-1]]
            current_entity = tokenizer.decode(
                sentence_token_ids[current_entity_index[0]: current_entity_index[1] + 1], skip_special_tokens=True)
            entities.append(current_entity)
            current_entity_index = [j]
            current_entity = None
        elif sentence_entity_predictions[j] == 1 and current_entity_index != []:
            current_entity_index.append(j)
        elif sentence_entity_predictions[j] == 1 and current_entity_index == []:
            continue
        elif sentence_entity_predictions[j] == 2 and current_entity_index != []:
            current_entity_index = [current_entity_index[0], current_entity_index[-1]]
            current_entity = tokenizer.decode(
                sentence_token_ids[current_entity_index[0]: current_entity_index[1] + 1], skip_special_tokens=True)
            entities.append(current_entity)
            current_entity_index = []
            current_entity = None
        elif sentence_entity_predictions[j] == 2 and current_entity_index == []:
            continue
        elif sentence_entity_predictions[j] == 3:
            break

entities = list(set(entities))
with open("../data/all_AMQP_entities", "wb") as file:
    pickle.dump(entities, file)