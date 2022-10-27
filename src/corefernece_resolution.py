from prepare_pretrain_data import prepare_pretrain_data
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import re
import numpy as np
from tqdm import tqdm
import pdfplumber
import nltk


class MeditationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def train(batch, model, optimizer):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_positions = batch["start_positions"].to(device)
    end_positions = batch["end_positions"].to(device)
    train_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                          start_positions=start_positions, end_positions=end_positions)
    train_loss = train_outputs.loss
    train_loss.backward()
    optimizer.step()
    return train_loss


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_positions = batch["start_positions"].to(device)
    end_positions = batch["end_positions"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         start_positions=start_positions, end_positions=end_positions)
    test_loss = test_outputs.loss
    return test_loss


rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")
rfc7959 = prepare_pretrain_data("rfc7959.txt", "Bormann & Shelby", "RFC 7959")
rfc8613 = prepare_pretrain_data("rfc8613.txt", "Selander, et al.", "RFC 8613")
rfc8974 = prepare_pretrain_data("rfc8974.txt", "?", "?")

# MQTT spec is a pdf file
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

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
# TODO: Need to include some "such" cases but need to specify them
PRONOUNS = ["it", "its", "they", "their", "them", "this field", "the field", "this value", "the value", "the packet"]

# rfc7252_rule_sentences = []
# for sentence in rfc7252:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             rfc7252_rule_sentences.append(sentence)
#             break
# rfc7252_rule_sentences = rfc7252_rule_sentences[1:]
#
# rfc7959_rule_sentences = []
# for sentence in rfc7959:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             rfc7959_rule_sentences.append(sentence)
#             break
#
# rfc8613_rule_sentences = []
# for sentence in rfc8613:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             rfc8613_rule_sentences.append(sentence)
#             break
#
# rfc8974_rule_sentences = []
# for sentence in rfc8974:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             rfc8974_rule_sentences.append(sentence)
#             break
#
# mqtt_rule_sentences = []
# for sentence in mqtt_spec_sentences:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             mqtt_rule_sentences.append(sentence)
#             break
# mqtt_rule_sentences = mqtt_rule_sentences[1:]
#
rfc7252_pronoun_sentences_pairs = []
for sentence in rfc7252:
    for pronoun in PRONOUNS:
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc7252_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))

        pronoun = pronoun.capitalize()
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc7252_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))

rfc7959_pronoun_sentences_pairs = []
for sentence in rfc7959:
    for pronoun in PRONOUNS:
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc7959_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))

        pronoun = pronoun.capitalize()
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc7959_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))
#
rfc8613_pronoun_sentences_pairs = []
for sentence in rfc8613:
    for pronoun in PRONOUNS:
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc8613_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))

        pronoun = pronoun.capitalize()
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc8613_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))
#
rfc8974_pronoun_sentences_pairs = []
for sentence in rfc8974:
    for pronoun in PRONOUNS:
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc8974_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))

        pronoun = pronoun.capitalize()
        # Regular expression for word boundary
        if re.search(r"\b" + pronoun + r"\b", sentence):
            iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
            start_indices = [m.start() for m in iter]
            for i in range(len(start_indices)):
                rfc8974_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))


# mqtt_pronoun_sentences_pairs = []
# for sentence in mqtt_spec_sentences:
#     for pronoun in PRONOUNS:
#         # Regular expression for word boundary
#         if re.search(r"\b" + pronoun + r"\b", sentence):
#             iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
#             start_indices = [m.start() for m in iter]
#             for i in range(len(start_indices)):
#                 mqtt_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))
#
#         pronoun = pronoun.capitalize()
#         # Regular expression for word boundary
#         if re.search(r"\b" + pronoun + r"\b", sentence):
#             iter = re.finditer(r"\b" + pronoun + r"\b", sentence)
#             start_indices = [m.start() for m in iter]
#             for i in range(len(start_indices)):
#                 mqtt_pronoun_sentences_pairs.append((sentence, f"{pronoun} number {i + 1}"))


def construct_context(pronoun_sentence, specification_sentences, k):
    pronoun_sentence_index = specification_sentences.index(pronoun_sentence)
    context_start_index = pronoun_sentence_index - k
    context_sentences = specification_sentences[context_start_index:pronoun_sentence_index + 1]
    return " ".join(context_sentences)


#
#
rfc7252_contexts = []
k = 5
for i in range(len(rfc7252_pronoun_sentences_pairs)):
    context = construct_context(rfc7252_pronoun_sentences_pairs[i][0], rfc7252, k)
    rfc7252_contexts.append((context, rfc7252_pronoun_sentences_pairs[i][1]))
#
rfc7959_contexts = []
k = 5
for i in range(len(rfc7959_pronoun_sentences_pairs)):
    context = construct_context(rfc7959_pronoun_sentences_pairs[i][0], rfc7959, k)
    rfc7959_contexts.append((context, rfc7959_pronoun_sentences_pairs[i][1]))
#
rfc8613_contexts = []
k = 5
for i in range(len(rfc8613_pronoun_sentences_pairs)):
    context = construct_context(rfc8613_pronoun_sentences_pairs[i][0], rfc8613, k)
    rfc8613_contexts.append((context, rfc8613_pronoun_sentences_pairs[i][1]))
#
rfc8974_contexts = []
k = 5
for i in range(len(rfc8974_pronoun_sentences_pairs)):
    context = construct_context(rfc8974_pronoun_sentences_pairs[i][0], rfc8974, k)
    rfc8974_contexts.append((context, rfc8974_pronoun_sentences_pairs[i][1]))
#
# mqtt_contexts = []
# k = 5
# for i in range(len(mqtt_pronoun_sentences_pairs)):
#     context = construct_context(mqtt_pronoun_sentences_pairs[i][0], mqtt_spec_sentences, k)
#     mqtt_contexts.append((context, mqtt_pronoun_sentences_pairs[i][1]))
#
data = []
for context in rfc7252_contexts:
    pronoun = context[1].strip().split("number")[0].strip()
    number = context[1].strip().split("number")[1].strip()
    data.append([context[0], f"What does '{pronoun}' number {number} refer to?"])
#
for context in rfc7959_contexts:
    pronoun = context[1].strip().split("number")[0].strip()
    number = context[1].strip().split("number")[1].strip()
    data.append([context[0], f"What does '{pronoun}' number {number} refer to?"])
#
for context in rfc8613_contexts:
    pronoun = context[1].strip().split("number")[0].strip()
    number = context[1].strip().split("number")[1].strip()
    data.append([context[0], f"What does '{pronoun}' number {number} refer to?"])

for context in rfc8974_contexts:
    pronoun = context[1].strip().split("number")[0].strip()
    number = context[1].strip().split("number")[1].strip()
    data.append([context[0], f"What does '{pronoun}' number {number} refer to?"])
#
# for context in mqtt_contexts:
#     data.append([context[0], f"What does '{context[1].strip()}' refer to?"])
# # data = data[:34]
#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForQuestionAnswering.from_pretrained("../model/iot_bert")
inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y = []
# y.append([94, 94])
# y.append([182, 183])
# y.append([155, 156])
# y.append([155, 156])
# y.append([11, 13])
# y.append([95, 95])
# # This is to indicate that the answer is not in the context
# y.append([0, 0])
# y.append([1, 3])
# y.append([96, 96])
# y.append([123, 124])
# y.append([214, 217])
# y.append([229, 230])
# y.append([0, 0])
# y.append([111, 111])
# y.append([218, 218])
# y.append([170, 202])
# y.append([193, 193])
# y.append([0, 0])
# y.append([149, 158])
# y.append([215, 217])
# y.append([185, 195])
# y.append([143, 143])
# y.append([196, 196])
# y.append([190, 191])
# y.append([194, 194])
# y.append([206, 206])
# y.append([159, 162])
# y.append([0, 0])
# y.append([255, 256])
# y.append([146, 147])
# y.append([224, 232])
# y.append([237, 238])
# y.append([282, 283])
# y.append([304, 304])
# y.append([115, 115])
# y.append([103, 103])
# y.append([133, 134])
# y.append([124, 126])
# y.append([124, 126])
# y.append([193, 194])
# y.append([196, 196])
# y.append([142, 149])
# y.append([228, 228])
# y.append([228, 228])
# y.append([178, 178])
# y.append([230, 233])
# y.append([169, 169])
# y.append([166, 169])
# y.append([164, 168])
# y.append([135, 136])
# y.append([126, 135])
# y.append([126, 135])
# y.append([188, 189])
# y.append([209, 212])
# y.append([238, 246])
# y.append([233, 235])
# y.append([233, 235])
# y.append([146, 147])
# y.append([253, 253])
# y.append([225, 226])
# y.append([0, 0])
# y.append([0, 0])
# y.append([186, 187])
# y.append([175, 177])
# y.append([109, 112])
# y.append([105, 108])
# y.append([195, 196])
# y.append([169, 170])
# y.append([185, 186])
# y.append([151, 155])
# y.append([155, 156])
# y.append([218, 218])
# y.append([168, 171])
# y.append([182, 183])
# y.append([164, 165])
# y.append([269, 273])
# y.append([206, 210])
# y.append([0, 0])
# y.append([179, 180])
# y.append([138, 142])
# y.append([161, 162])
# y.append([144, 145])
# y.append([151, 151])
# y.append([118, 118])
# y.append([121, 121])
# y.append([141, 141])
# y.append([144, 144])
# y.append([123, 123])
# y.append([0, 0])
# y.append([0, 0])
# y.append([153, 153])
# y.append([193, 194])
# y.append([161, 164])
# y.append([151, 151])
# y.append([213, 214])
# y.append([175, 177])
# y.append([241, 243])
# y.append([133, 135])
# y.append([133, 135])
# y.append([120, 121])
# y.append([112, 112])
#
# print(len(y))

# y = np.array(y)
# y_start = y[:, 0].tolist()
# y_end = y[:, 1].tolist()
#
# inputs["start_positions"] = torch.tensor(y_start)
# inputs["end_positions"] = torch.tensor(y_end)
#
# dataset = MeditationDataset(inputs)
# dataset_length = len(dataset)
# train_length = int(dataset_length * 0.9)
# test_length = dataset_length - train_length
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#
# for epoch in range(20):
#     train_loop = tqdm(train_loader, leave=True)
#     overall_train_loss = 0
#     num_of_train_batches = len(train_loader)
#     for train_batch in train_loop:
#         model.train()
#         train_loss = train(train_batch, model, optimizer)
#         train_loop.set_postfix(train_loss=train_loss.item())
#         overall_train_loss += train_loss.item()
#         train_loop.set_description(f"Epoch {epoch} train")
#
#     test_loop = tqdm(test_loader, leave=True)
#     overall_test_loss = 0
#     num_of_test_batches = len(test_loader)
#     for test_batch in test_loop:
#         model.eval()
#         test_loss = test(test_batch, model)
#         test_loop.set_postfix(test_loss=test_loss.item())
#         overall_test_loss += test_loss.item()
#         test_loop.set_description(f"Epoch {epoch} test")
#
#     average_train_loss = overall_train_loss / num_of_train_batches
#     print(f"average train loss: {average_train_loss}")
#     average_test_loss = overall_test_loss / num_of_test_batches
#     print(f"average test loss: {average_test_loss}")
#
#     with open(r"../results/coref_resolution.txt", "a") as file:
#         file.write(
#             f"Epoch {epoch} average_train_loss: {average_train_loss}")
#         file.write("\n")
#         file.write(
#             f"Epoch {epoch} average_test_loss: {average_test_loss}")
#         file.write("\n")
