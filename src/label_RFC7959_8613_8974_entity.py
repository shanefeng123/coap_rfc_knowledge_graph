from prepare_pretrain_data import prepare_pretrain_data
import random
import torch
from transformers import BertTokenizer
from sklearn import metrics

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
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


# 7959
# rfc7959 = prepare_pretrain_data("rfc7959.txt", "Bormann & Shelby", "RFC 7959")
# rule_sentences_7959 = []
# # 43 rule sentences in RFC7959 in total, NO Sampling
# for sentence in rfc7959:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             rule_sentences_7959.append(sentence)
#             break
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# inputs = tokenizer(rule_sentences_7959, padding="max_length", truncation=True, return_tensors="pt")
# input_ids = inputs["input_ids"]
# sentence_tokens = []
# for ids in input_ids:
#     sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))
#
# y = []
# y.append(annotate_entity(sentence_tokens[0], [(16, 16), (28, 28)]))
# y.append(annotate_entity(sentence_tokens[1], [(119, 120)]))
# y.append(annotate_entity(sentence_tokens[2], [(9, 11), (22, 22)]))
# y.append(annotate_entity(sentence_tokens[3], [(2, 3), (16, 16)]))
# y.append(annotate_entity(sentence_tokens[4], [(3, 6), (11, 11), (16, 17), (26, 27)]))
# y.append(annotate_entity(sentence_tokens[5], [(5, 7), (53, 53)]))
# y.append(annotate_entity(sentence_tokens[6], [(3, 4), (7, 9), (18, 18), (25, 26)]))
# y.append(annotate_entity(sentence_tokens[7], [(2, 3), (7, 7), (13, 15), (19, 19), (24, 26), (34, 34)]))
# y.append(annotate_entity(sentence_tokens[8], [(7, 8)]))
# y.append(annotate_entity(sentence_tokens[9], [(6, 6), (14, 17)]))
# y.append(annotate_entity(sentence_tokens[10],
#                          [(5, 7), (8, 9), (14, 17), (21, 22), (26, 26), (48, 48), (55, 56), (64, 64), (76, 77),
#                           (85, 86)]))  #
# y.append(annotate_entity(sentence_tokens[11], [(2, 7), (11, 11), (13, 13), (19, 22)]))
# y.append(annotate_entity(sentence_tokens[12], [(2, 2), (5, 5), (9, 9), (15, 20), (23, 23), (34, 34), (66, 70)]))
# y.append(annotate_entity(sentence_tokens[13], [(3, 3), (6, 6), (12, 13), (16, 19), (24, 24)]))
# y.append(annotate_entity(sentence_tokens[14], [(4, 5), (9, 9), (13, 13), (19, 22), (26, 26), (34, 35), (40, 41)]))
# y.append(annotate_entity(sentence_tokens[15], [(2, 2), (8, 9)]))
# y.append(annotate_entity(sentence_tokens[16], [(3, 6), (8, 8), (19, 20), (26, 26), (29, 29), (33, 33), (41, 44)]))
# y.append(annotate_entity(sentence_tokens[17], [(2, 2), (5, 6), (29, 30), (34, 34), (40, 40), (48, 48)]))
# y.append(annotate_entity(sentence_tokens[18], [(16, 17), (20, 20), (32, 33), (50, 53)]))
# y.append(annotate_entity(sentence_tokens[19], [(7, 10), (21, 24), (45, 45), (61, 61), (68, 71), (74, 74)]))
# y.append(annotate_entity(sentence_tokens[20], [(3, 6), (11, 11), (22, 22), (30, 33)]))
# y.append(annotate_entity(sentence_tokens[21], [(10, 10), (25, 25)]))
# y.append(annotate_entity(sentence_tokens[22], [(1, 3), (9, 9)]))
# y.append(annotate_entity(sentence_tokens[23], [(4, 4), (19, 19), (23, 24), (28, 28)]))
# y.append(annotate_entity(sentence_tokens[24], [(10, 10), (21, 21)]))
# y.append(annotate_entity(sentence_tokens[25], [(2, 2), (22, 24), (33, 33)]))
# y.append(annotate_entity(sentence_tokens[26], [(1, 3), (17, 17)]))
# y.append(annotate_entity(sentence_tokens[27],
#                          [(3, 3), (19, 20), (30, 31), (33, 33), (37, 38), (43, 43), (61, 61), (65, 66), (73, 73)]))
# y.append(annotate_entity(sentence_tokens[28], [(13, 13), (26, 26), (29, 29), (36, 39), (42, 42), (50, 50), (56, 59)]))
# y.append(annotate_entity(sentence_tokens[29], [(10, 11), (13, 13), (19, 19), (31, 31)]))
# y.append(annotate_entity(sentence_tokens[30], [(1, 2), (11, 11), (14, 15)]))
# y.append(annotate_entity(sentence_tokens[31],
#                          [(6, 6), (11, 11), (23, 23), (26, 26), (34, 35), (40, 40), (43, 43), (48, 48)]))
# y.append(annotate_entity(sentence_tokens[32], [(12, 12), (19, 23), (31, 32), (37, 37), (47, 47)]))
# y.append(annotate_entity(sentence_tokens[33], [(1, 5), (10, 12), (15, 16), (29, 31), (33, 34)]))
# y.append(annotate_entity(sentence_tokens[34], [(6, 6), (15, 16), (28, 29)]))
# y.append(annotate_entity(sentence_tokens[35], [(18, 18), (21, 24), (32, 32)]))
# y.append(annotate_entity(sentence_tokens[36], [(2, 4), (17, 17), (25, 25)]))
# y.append(annotate_entity(sentence_tokens[37], [(2, 4), (17, 17)]))
# y.append(annotate_entity(sentence_tokens[38], [(13, 17), (19, 20), (34, 35), (43, 44), (47, 48)]))
# y.append(annotate_entity(sentence_tokens[39], []))
# y.append(annotate_entity(sentence_tokens[40], [(1, 2)]))
# y.append(annotate_entity(sentence_tokens[41], [(1, 2)]))
# y.append(annotate_entity(sentence_tokens[42], [(9, 9), (18, 21)]))


#####################################   RFC8613   #########################################################
# rfc8613 = prepare_pretrain_data("rfc8613.txt", "Selander, et al.", "RFC 8613")
# rule_sentences_8613 = []
# for sentence in rfc8613:
#     for keyword in MODAL_KEYWORDS:
#         if keyword in sentence:
#             rule_sentences_8613.append(sentence)
#             break
# # 147 rule sentences in RFC8613, DO Sampling
# random.seed(4)
# num_of_samples_8613 = round(len(rule_sentences_8613) * 0.2)
# sampled_8613_rule_sentences = random.sample(rule_sentences_8613, num_of_samples_8613)
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# inputs = tokenizer(sampled_8613_rule_sentences, padding="max_length", truncation=True, return_tensors="pt")
# input_ids = inputs["input_ids"]
# sentence_tokens = []
# for ids in input_ids:
#     sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))
#
# y1 = []
# y1.append(annotate_entity(sentence_tokens[0], [(6, 8), (12, 15), (16, 16), (22, 25), (36, 38)]))
# y1.append(annotate_entity(sentence_tokens[1], [(26, 33), (35, 36)]))
# y1.append(annotate_entity(sentence_tokens[2], [(13, 16), (18, 21), (24, 24), (26, 26), (39, 42), (50, 50)]))
# y1.append(annotate_entity(sentence_tokens[3], [(2, 2), (9, 13), (18, 20), (25, 26)]))
# y1.append(annotate_entity(sentence_tokens[4], [(4, 4), (9, 14), (16, 20), (31, 31)]))
# y1.append(annotate_entity(sentence_tokens[5], [(7, 7)]))
# y1.append(annotate_entity(sentence_tokens[6], [(8, 9), (16, 18), (36, 38)]))
# y1.append(annotate_entity(sentence_tokens[7], [(6, 8), (19, 21)]))
# y1.append(annotate_entity(sentence_tokens[8], [(1, 2), (8, 8), (10, 10)]))
# y1.append(annotate_entity(sentence_tokens[9], [(8, 8), (15, 15)]))
# y1.append(annotate_entity(sentence_tokens[10], [(3, 4), (10, 10), (15, 18), (21, 21), (28, 30)]))
# y1.append(annotate_entity(sentence_tokens[11], [(1, 2)]))
# y1.append(annotate_entity(sentence_tokens[12], [(6, 9)]))
# y1.append(annotate_entity(sentence_tokens[13], [(14, 14), (21, 21)]))
# y1.append(annotate_entity(sentence_tokens[14], []))
# y1.append(annotate_entity(sentence_tokens[15], [(3, 4), (10, 10), (14, 16), (23, 24)]))
# y1.append(annotate_entity(sentence_tokens[16], [(2, 2), (9, 11), (18, 18), (24, 26), (30, 32), (40, 40), (42, 44)]))
# y1.append(annotate_entity(sentence_tokens[17], [(2, 2), (8, 9)]))
# y1.append(annotate_entity(sentence_tokens[18], [(2, 2), (24, 26), (36, 36)]))
# y1.append(annotate_entity(sentence_tokens[19], [(1, 2), (5, 5), (8, 11), (14, 16), (24, 25), (40, 42)]))
# y1.append(annotate_entity(sentence_tokens[20], [(2, 2), (7, 10)]))
# y1.append(annotate_entity(sentence_tokens[21], [(10, 12), (20, 20), (25, 30), (32, 36), (48, 48), (55, 55)]))
# y1.append(annotate_entity(sentence_tokens[22], [(2, 3), (6, 8), (10, 10), (15, 18)]))
# y1.append(annotate_entity(sentence_tokens[23], [(7, 9)]))
# y1.append(annotate_entity(sentence_tokens[24], [(2, 4), (12, 14), (22, 24)]))
# y1.append(annotate_entity(sentence_tokens[25], [(4, 4), (9, 12), (20, 21), (25, 25)]))
# y1.append(annotate_entity(sentence_tokens[26], [(6, 7), (14, 16), (22, 24)]))
# y1.append(annotate_entity(sentence_tokens[27], [(2, 7), (20, 22)]))
# y1.append(annotate_entity(sentence_tokens[28], [(3, 6), (12, 15), (18, 20)]))


#####################################   RFC8974   #########################################################
rfc8974 = prepare_pretrain_data("rfc8974.txt", "?", "?")
rule_sentences_8974 = []
for sentence in rfc8974:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rule_sentences_8974.append(sentence)
            break

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(rule_sentences_8974, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y2 = []
y2.append(annotate_entity(sentence_tokens[0], []))
y2.append(annotate_entity(sentence_tokens[1], []))
y2.append(annotate_entity(sentence_tokens[2], []))
y2.append(annotate_entity(sentence_tokens[3], []))
y2.append(annotate_entity(sentence_tokens[4], []))
y2.append(annotate_entity(sentence_tokens[5], [(3, 3), (6, 6), (9, 9), (20, 27), (35, 35)]))
y2.append(annotate_entity(sentence_tokens[6], [(3, 3), (6, 6), (9, 9), (25, 32), (41, 41), (54, 54)]))
y2.append(annotate_entity(sentence_tokens[7], [(20, 20)]))
y2.append(annotate_entity(sentence_tokens[8], [(7, 9), (21, 21), (24, 28)]))
y2.append(annotate_entity(sentence_tokens[9], [(19, 19), (34, 36)]))
y2.append(annotate_entity(sentence_tokens[10], [(2, 3), (8, 8), (17, 19), (24, 24)]))
y2.append(annotate_entity(sentence_tokens[11], [(3, 3), (5, 7), (11, 11), (14, 14), (34, 34), (41, 43)]))
y2.append(annotate_entity(sentence_tokens[12], [(5, 5), (9, 9), (31, 31), (35, 35), (43, 43)]))
y2.append(annotate_entity(sentence_tokens[13], [(18, 18), (34, 34)]))
y2.append(annotate_entity(sentence_tokens[14], [(6, 6), (12, 13), (22, 23), (26, 27)]))
y2.append(annotate_entity(sentence_tokens[15], [(2, 3), (5, 6)]))
y2.append(annotate_entity(sentence_tokens[16], [(5, 5), (21, 21), (25, 25), (31, 32)]))
y2.append(annotate_entity(sentence_tokens[17], [(2, 2), (14, 14)]))
y2.append(annotate_entity(sentence_tokens[18], [(4, 4), (12, 12), (40, 41), (50, 50)]))
y2.append(annotate_entity(sentence_tokens[19], [(20, 20), (23, 23)]))
y2.append(annotate_entity(sentence_tokens[20], []))
y2.append(annotate_entity(sentence_tokens[21], [(2, 4), (18, 22), (40, 40), (43, 43), (57, 59), (62, 62), (75, 76)]))
y2.append(annotate_entity(sentence_tokens[22], [(5, 5), (21, 21)]))
y2.append(annotate_entity(sentence_tokens[23], [(4, 5), (14, 18), (20, 20), (31, 31)]))#
y2.append(annotate_entity(sentence_tokens[24], [(1, 3), (21, 22), (38, 40), (48,48),(51, 52)]))
y2.append(annotate_entity(sentence_tokens[25], [(1, 3), (21, 22)]))
y2.append(annotate_entity(sentence_tokens[26], []))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cpu":
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")
model = torch.load("../model/entity_extractor.pt", map_location=device)

labels = torch.LongTensor(y2)
inputs["labels"] = labels

model.to(device)
model.eval()
test_loss, predictions, accuracy, labels = test(inputs, model)
print(metrics.classification_report(torch.flatten(labels).tolist(), torch.flatten(predictions).tolist(), zero_division=0))
print(accuracy.item())