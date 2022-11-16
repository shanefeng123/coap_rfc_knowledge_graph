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
    train_predict_start_positions = torch.argmax(train_outputs.start_logits, dim=-1)
    train_predict_end_positions = torch.argmax(train_outputs.end_logits, dim=-1)
    train_loss.backward()
    optimizer.step()
    return train_loss, train_predict_start_positions, train_predict_end_positions, start_positions, end_positions


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    start_positions = batch["start_positions"].to(device)
    end_positions = batch["end_positions"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         start_positions=start_positions, end_positions=end_positions)
    test_predict_start_positions = torch.argmax(test_outputs.start_logits, dim=-1)
    test_predict_end_positions = torch.argmax(test_outputs.end_logits, dim=-1)
    test_loss = test_outputs.loss
    return test_loss, test_predict_start_positions, test_predict_end_positions, start_positions, end_positions


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
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertForQuestionAnswering.from_pretrained("bert-large-cased")
inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y = []
# By yansong
# 0
y.append([233, 234])
y.append([99, 101])
y.append([90, 92])
y.append([181, 181])
y.append([198, 201])
y.append([162, 174])
y.append([164, 168])
y.append([132, 134])
y.append([126, 126])
y.append([173, 178])
# 10
y.append([173, 178])
y.append([92, 93])
y.append([156, 156])
y.append([223, 223])
y.append([167, 168])
y.append([180, 180])
y.append([212, 212])
y.append([0, 0])
y.append([195, 196])
y.append([0, 0])
# 20
y.append([135, 136])
y.append([94, 98])
y.append([102, 103])
y.append([111, 112])
y.append([156, 156])
y.append([165, 167])
y.append([11, 13])
y.append([95, 95])
y.append([120, 121])
y.append([105, 111])

# 30
y.append([1, 3])
y.append([96, 96])
y.append([123, 124])
y.append([161, 163])
y.append([116, 118])
y.append([69, 74])
y.append([119, 120])
y.append([118, 122])
y.append([118, 122])
y.append([193, 197])
# 40
y.append([211, 220])
y.append([187, 189])
y.append([214, 217])
y.append([235, 236])
y.append([198, 199])
y.append([0, 0])
y.append([229, 230])
y.append([0, 0])
y.append([260, 264])
y.append([111, 111])
# 50
y.append([163, 165])
y.append([0, 0])
y.append([230, 230])
y.append([288, 288])
y.append([0, 0])
y.append([0, 0])
y.append([240, 242])
y.append([0, 0])
y.append([300, 300])
y.append([250, 250])
# 60
y.append([218, 218])
y.append([170, 172])
y.append([193, 193])
y.append([193, 193])
y.append([0, 0])
y.append([0, 0])
y.append([149, 158])
y.append([327, 331])
y.append([294, 296])
y.append([262, 272])
# 70
y.append([314, 324])
y.append([403, 404])
y.append([403, 404])
y.append([253, 254])
y.append([134, 142])
y.append([408, 412])
y.append([356, 363])
y.append([0, 0])
y.append([0, 0])
y.append([178, 179])
# 80
y.append([146, 159])
y.append([146, 159])
y.append([214, 217])
y.append([184, 195])
y.append([183, 185])
y.append([183, 185])
y.append([202, 206])
y.append([112, 115])
y.append([0, 0])
y.append([142, 142])
# 90
y.append([198, 198])
y.append([182, 183])
y.append([0, 0])
y.append([123, 124])
y.append([92, 93])
y.append([136, 137])
y.append([142, 143])
y.append([195, 196])
y.append([271, 271])
y.append([175, 175])
# 100
y.append([118, 118])
y.append([130, 131])
y.append([169, 170])
y.append([164, 165])
y.append([190, 191])
y.append([194, 194])
y.append([222, 222])
y.append([187, 189])
y.append([133, 145])
y.append([206, 206])
# 110
y.append([213, 213])
y.append([234, 234])
y.append([210, 211])
y.append([217, 217])
y.append([178, 178])
y.append([134, 134])
y.append([134, 134])
y.append([0, 0])
y.append([0, 0])
y.append([140, 154])
# 120
y.append([175, 175])
y.append([196, 196])
y.append([196, 196])
y.append([169, 169])
y.append([123, 123])
y.append([140, 153])
y.append([0, 0])
y.append([103, 104])
y.append([103, 104])
y.append([166, 166])
# 130
y.append([0, 0])
y.append([152, 152])
y.append([152, 152])
y.append([179, 179])
y.append([0, 0])
y.append([191, 197])
y.append([274, 275])
y.append([266, 266])
y.append([296, 296])
y.append([137, 141])
# 140
y.append([165, 165])
y.append([146, 147])
y.append([137, 138])
y.append([194, 195])
y.append([143, 144])
y.append([143, 144])
y.append([208, 209])
y.append([160, 162])
y.append([140, 140])
y.append([220, 221])
# 150
y.append([220, 221])
y.append([220, 221])
y.append([193, 193])
y.append([191, 192])
y.append([119, 120])
y.append([119, 120])
y.append([119, 120])
y.append([153, 154])
y.append([193, 193])
y.append([224, 232])
# 160
y.append([142, 150])
y.append([237, 238])
y.append([282, 283])
y.append([309, 309])
y.append([277, 282])
y.append([279, 282])
y.append([285, 286])
y.append([285, 286])
y.append([189, 192])
y.append([159, 162])
# 170
y.append([181, 181])
y.append([80, 83])
y.append([115, 115])
y.append([0, 0])
y.append([115, 116])
y.append([103, 103])
y.append([188, 188])
y.append([115, 116])
y.append([0, 0])
y.append([104, 106])
# 180
y.append([90, 90])
y.append([129, 129])
y.append([133, 134])
y.append([151, 151])
y.append([55, 57])
y.append([0, 0])
y.append([149, 151])
y.append([73, 74])
y.append([73, 74])
y.append([124, 126])
# 190
y.append([124, 126])
y.append([0, 0])
y.append([191, 194])
y.append([241, 242])
y.append([0, 0])
y.append([0, 0])
y.append([166, 173])
y.append([0, 0])
y.append([59, 63])
y.append([47, 51])
# 200
y.append([182, 183])
y.append([193, 194])
y.append([152, 156])
y.append([114, 115])
y.append([81, 83])
y.append([0, 0])
y.append([0, 0])
y.append([182, 192])
y.append([182, 192])
y.append([256, 257])
# 210
y.append([104, 108])
y.append([104, 108])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
y.append([221, 222])
y.append([196, 196])
y.append([206, 211])
y.append([153, 153])
y.append([204, 207])
# 220
y.append([216, 316])  #
y.append([190, 192])
y.append([158, 160])
y.append([158, 160])
y.append([0, 0])
y.append([110, 116])
y.append([70, 76])
y.append([130, 133])
y.append([142, 149])
y.append([228, 228])
# 230
y.append([174, 174])
y.append([174, 174])
y.append([190, 190])
y.append([190, 190])
y.append([140, 140])
y.append([120, 120])
y.append([120, 120])
y.append([230, 233])
y.append([303, 303])
y.append([169, 169])
# 240
y.append([158, 158])
y.append([181, 183])
y.append([131, 132])
y.append([212, 213])
y.append([218, 219])
y.append([218, 219])
y.append([174, 181])
y.append([0, 0])
y.append([0, 0])
y.append([262, 264])
# 250
y.append([110, 111])
y.append([142, 142])
y.append([169, 172])
y.append([143, 144])
y.append([143, 144])
y.append([166, 169])
y.append([167, 168])
y.append([184, 184])
y.append([213, 214])
y.append([210, 210])
# 260
y.append([0, 0])
y.append([135, 136])
y.append([134, 137])
y.append([172, 176])
y.append([160, 162])
y.append([0, 0])
y.append([221, 223])
y.append([169, 170])
y.append([207, 210])
y.append([128, 130])

# 270
y.append([116, 119])
y.append([145, 145])
y.append([191, 195])
y.append([188, 190])
y.append([160, 162])
y.append([125, 127])
y.append([0, 0])
y.append([144, 152])
y.append([161, 161])
y.append([175, 177])
# 280
y.append([0, 0])
y.append([167, 167])
y.append([0, 0])
y.append([188, 188])
y.append([229, 229])
y.append([172, 173])
y.append([172, 176])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
# 290
y.append([151, 154])
y.append([82, 85])
y.append([126, 135])
y.append([106, 107])
y.append([121, 122])
y.append([126, 135])
y.append([0, 0])
y.append([0, 0])
y.append([86, 88])
y.append([112, 113])
# 300
y.append([0, 0])
y.append([197, 199])
y.append([253, 255])
y.append([249, 249])
y.append([62, 64])
y.append([152, 157])
y.append([279, 182])
y.append([183, 193])
y.append([201, 203])
y.append([206, 207])
# 310
y.append([198, 210])
y.append([2, 6])
y.append([203, 208])
y.append([187, 189])
y.append([187, 189])
y.append([187, 189])
y.append([0, 0])  # the value
y.append([0, 0])  # the value
y.append([0, 0])  # the value
y.append([179, 180])
# 320
y.append([0, 0])  # the value
y.append([159, 166])
y.append([0, 0])  # the value
y.append([203, 203])
y.append([221, 228])
y.append([239, 240])
y.append([239, 243])
y.append([266, 269])
y.append([0, 0])
y.append([213, 214])
# 330
y.append([209, 209])
y.append([243, 246])
y.append([233, 235])
y.append([244, 245])
y.append([233, 235])
y.append([272, 274])
y.append([0, 0])
y.append([278, 283])
y.append([274, 277])
y.append([209, 210])
# 340
y.append([148, 149])
y.append([146, 147])
y.append([162, 164])
y.append([180, 181])
y.append([253, 253])
y.append([175, 176])
y.append([148, 149])
y.append([148, 149])
y.append([148, 149])
y.append([117, 117])
# 350
y.append([117, 117])
y.append([108, 108])
y.append([108, 108])
y.append([234, 235])
y.append([234, 235])
y.append([189, 191])
y.append([255, 256])
y.append([223, 224])
y.append([238, 240])
y.append([172, 173])
# 360
y.append([131, 132])
y.append([169, 170])
y.append([195, 196])
y.append([168, 172])
y.append([225, 226])
y.append([225, 226])
y.append([194, 195])
y.append([0, 0])
y.append([125, 127])
y.append([144, 149])
# 370
y.append([205, 206])
y.append([0, 0])  # the value
y.append([113, 116])
y.append([152, 163])
y.append([261, 261])
y.append([214, 214])
y.append([0, 0])
y.append([0, 0])
y.append([263, 265])
y.append([224, 227])
# 380
y.append([0, 0])
y.append([186, 187])
y.append([181, 183])
y.append([0, 0])
y.append([198, 200])
y.append([148, 150])
y.append([182, 184])
y.append([175, 177])
y.append([175, 177])
y.append([127, 129])
# 390
y.append([154, 156])
y.append([154, 156])
y.append([174, 177])
y.append([183, 185])
y.append([83, 96])
y.append([126, 128])
y.append([126, 128])
y.append([0, 0])  # the value
y.append([291, 292])
y.append([141, 143])
# 400
y.append([0, 0])  # the value
y.append([110, 112])
y.append([105, 108])
y.append([210, 211])
y.append([147, 155])
y.append([147, 155])
y.append([0, 0])
y.append([360, 369])
y.append([199, 202])
y.append([215, 215])
# 410
y.append([0, 0])  # the value
y.append([169, 170])
y.append([210, 212])
y.append([185, 186])
y.append([150, 155])
y.append([0, 0])  # the value
y.append([155, 156])
y.append([157, 158])
y.append([172, 173])
y.append([188, 189])
# 420
y.append([198, 201])
y.append([237, 239])
y.append([168, 171])
y.append([142, 145])
y.append([181, 183])
y.append([25, 27])
y.append([161, 162])
y.append([152, 158])
y.append([138, 142])
y.append([0, 0])
# 430
y.append([156, 157])
y.append([0, 0])
y.append([270, 273])
y.append([0, 0])
y.append([0, 0])
y.append([173, 181])
y.append([173, 181])
y.append([145, 149])
y.append([149, 153])
y.append([0, 0])
# 440
y.append([0, 0])
y.append([206, 210])
y.append([162, 162])
y.append([0, 0])
y.append([174, 176])
y.append([180, 180])
y.append([173, 174])
y.append([138, 142])
y.append([161, 162])
y.append([173, 174])
# 450
y.append([147, 151])
y.append([124, 128])
y.append([250, 258])
y.append([240, 243])
y.append([269, 276])
y.append([261, 264])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
# 460
y.append([169, 171])
y.append([182, 186])
y.append([0, 0])
y.append([121, 124])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
y.append([98, 99])
y.append([0, 0])  # not found
y.append([0, 0])
# 470
y.append([0, 0])
y.append([0, 0])
y.append([150, 151])
y.append([0, 0])
y.append([124, 126])
y.append([124, 126])
y.append([167, 168])
y.append([196, 196])
y.append([196, 196])
y.append([196, 196])
# 480
y.append([182, 184])
y.append([151, 151])
y.append([137, 139])
y.append([0, 0])
y.append([158, 158])
y.append([0, 0])
y.append([182, 189])
y.append([182, 189])
y.append([121, 121])
y.append([118, 118])
# 490
y.append([121, 121])
y.append([144, 144])
y.append([141, 141])
y.append([144, 144])
y.append([123, 123])
y.append([169, 170])
y.append([202, 202])
y.append([207, 207])
y.append([0, 0])
y.append([158, 160])
# 500
y.append([131, 133])
y.append([0, 0])
y.append([153, 153])
y.append([153, 153])
y.append([164, 164])
y.append([194, 194])
y.append([256, 256])
y.append([254, 254])
y.append([247, 247])
y.append([217, 217])

# 510
y.append([240, 241])
y.append([213, 215])
y.append([235, 237])
y.append([170, 170])
y.append([170, 170])
y.append([161, 161])
y.append([181, 181])
y.append([162, 164])
y.append([151, 151])
y.append([177, 177])
# 520
y.append([175, 175])
y.append([176, 178])
y.append([237, 237])
y.append([250, 250])
y.append([262, 264])
y.append([262, 264])
y.append([248, 250])
y.append([203, 205])
y.append([203, 205])
y.append([203, 205])
# 530
y.append([178, 180])
y.append([189, 190])
y.append([175, 177])
y.append([175, 177])
y.append([175, 177])
y.append([230, 230])
y.append([241, 143])
y.append([268, 270])
y.append([268, 270])
y.append([175, 197])
# 540
y.append([138, 140])
y.append([133, 135])
y.append([133, 135])
y.append([131, 133])
y.append([131, 133])
y.append([177, 178])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
y.append([0, 0])
# 550
y.append([0, 0])
y.append([121, 121])

#
# print(len(y))

y = np.array(y)
y_start = y[:, 0].tolist()
y_end = y[:, 1].tolist()

inputs["start_positions"] = torch.tensor(y_start)
inputs["end_positions"] = torch.tensor(y_end)

dataset = MeditationDataset(inputs)
dataset_length = len(dataset)
train_length = int(dataset_length * 0.9)
test_length = dataset_length - train_length
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(100):
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    epoch_train_predict_start_positions = None
    epoch_train_predict_end_positions = None
    epoch_train_start_positions = None
    epoch_train_end_positions = None
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        model.train()
        train_loss, train_predict_start_positions, train_predict_end_positions, train_start_positions, train_end_positions = train(
            train_batch, model, optimizer)
        train_accuracy = ((train_start_positions == train_predict_start_positions) * (
                train_end_positions == train_predict_end_positions)).int().sum() / train_start_positions.shape[0]
        train_loop.set_postfix(train_loss=train_loss.item(), train_accuracy=train_accuracy.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Epoch {epoch} train")

        if epoch_train_predict_start_positions is None:
            epoch_train_predict_start_positions = train_predict_start_positions
            epoch_train_predict_end_positions = train_predict_end_positions
            epoch_train_start_positions = train_start_positions
            epoch_train_end_positions = train_end_positions
        else:
            epoch_train_predict_start_positions = torch.cat(
                (epoch_train_predict_start_positions, train_predict_start_positions), dim=0)
            epoch_train_predict_end_positions = torch.cat(
                (epoch_train_predict_end_positions, train_predict_end_positions), dim=0)
            epoch_train_start_positions = torch.cat((epoch_train_start_positions, train_start_positions), dim=0)
            epoch_train_end_positions = torch.cat((epoch_train_end_positions, train_end_positions), dim=0)

    test_loop = tqdm(test_loader, leave=True)
    epoch_test_predict_start_positions = None
    epoch_test_predict_end_positions = None
    epoch_test_start_positions = None
    epoch_test_end_positions = None
    overall_test_loss = 0
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss, test_predict_start_positions, test_predict_end_positions, test_start_positions, test_end_positions = test(
            test_batch, model)
        test_accuracy = ((test_start_positions == test_predict_start_positions) * (
                test_end_positions == test_predict_end_positions)).int().sum() / test_start_positions.shape[0]
        test_loop.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Epoch {epoch} test")

        if epoch_test_predict_start_positions is None:
            epoch_test_predict_start_positions = test_predict_start_positions
            epoch_test_predict_end_positions = test_predict_end_positions
            epoch_test_start_positions = test_start_positions
            epoch_test_end_positions = test_end_positions
        else:
            epoch_test_predict_start_positions = torch.cat(
                (epoch_test_predict_start_positions, test_predict_start_positions), dim=0)
            epoch_test_predict_end_positions = torch.cat(
                (epoch_test_predict_end_positions, test_predict_end_positions), dim=0)
            epoch_test_start_positions = torch.cat((epoch_test_start_positions, test_start_positions), dim=0)
            epoch_test_end_positions = torch.cat((epoch_test_end_positions, test_end_positions), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    print(f"average train loss: {average_train_loss}")
    average_test_loss = overall_test_loss / num_of_test_batches
    print(f"average test loss: {average_test_loss}")

    epoch_train_accuracy = ((epoch_train_start_positions == epoch_train_predict_start_positions) * (
                epoch_train_end_positions == epoch_train_predict_end_positions)).int().sum() / epoch_train_start_positions.shape[0]
    print(f"epoch train accuracy: {epoch_train_accuracy}")
    epoch_test_accuracy = ((epoch_test_start_positions == epoch_test_predict_start_positions) * (
                epoch_test_end_positions == epoch_test_predict_end_positions)).int().sum() / epoch_test_start_positions.shape[0]
    print(f"epoch test accuracy: {epoch_test_accuracy}")

    with open(r"../results/coref_resolution.txt", "a") as file:
        file.write("\n")
        file.write(
            f"Epoch {epoch} average_train_loss: {average_train_loss}")
        file.write("\n")
        file.write(
            f"Epoch {epoch} average_test_loss: {average_test_loss}")
        file.write("\n")
        file.write(
            f"Epoch {epoch} train_accuracy: {epoch_train_accuracy}"
        )
        file.write("\n")
        file.write(
            f"Epoch {epoch} test_accuracy: {epoch_test_accuracy}"
        )
        file.write("\n")

    if epoch_test_accuracy > 0.7:
        break

torch.save(model, r"../model/coref")
