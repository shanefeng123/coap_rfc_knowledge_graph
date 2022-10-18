from prepare_pretrain_data import prepare_pretrain_data
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import re
import numpy as np
from tqdm import tqdm


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

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
# TODO: Need to include some "such" cases but need to specify them
PRONOUNS = ["It", "it", "Its", "its", "They", " they", "Their", "their", "Them", "them", "This field",
            "this field", "the field", "The field", "This value", "this value"]

rule_sentences = []
for sentence in rfc7252:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rule_sentences.append(sentence)
            break
rule_sentences = rule_sentences[1:]

pronoun_sentences_pairs = []
for sentence in rule_sentences:
    for pronoun in PRONOUNS:
        if re.search(r"\b" + pronoun + r"\b", sentence):
            pronoun_sentences_pairs.append((sentence, pronoun))


# pronoun_sentences_pairs = []
# for pronoun_sentence in pronoun_sentences:
#     for pronoun in PRONOUNS:
#         if pronoun in pronoun_sentence:
#             pronoun_sentences_pairs.append((pronoun_sentence, pronoun))


def construct_context(pronoun_sentence, specification_sentences, k):
    pronoun_sentence_index = specification_sentences.index(pronoun_sentence)
    context_start_index = pronoun_sentence_index - k
    context_sentences = specification_sentences[context_start_index:pronoun_sentence_index + 1]
    return " ".join(context_sentences)


contexts = []
k = 5
for i in range(len(pronoun_sentences_pairs)):
    context = construct_context(pronoun_sentences_pairs[i][0], rfc7252, k)
    contexts.append((context, pronoun_sentences_pairs[i][1]))

data = []
for context in contexts:
    data.append([context[0], f"What does '{context[1].strip()}' refer to?"])

# data = data[:34]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForQuestionAnswering.from_pretrained("../model/iot_bert")
inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

y = []
y.append([94, 94])
y.append([182, 183])
y.append([155, 156])
y.append([155, 156])
y.append([11, 13])
y.append([95, 95])
# This is to indicate that the answer is not in the context
y.append([0, 0])
y.append([1, 3])
y.append([96, 96])
y.append([123, 124])
y.append([214, 217])
y.append([229, 230])
y.append([0, 0])
y.append([111, 111])
y.append([218, 218])
y.append([170, 202])
y.append([193, 193])
y.append([0, 0])
y.append([149, 158])
y.append([215, 217])
y.append([185, 195])
y.append([143, 143])
y.append([196, 196])
y.append([190, 191])
y.append([194, 194])
y.append([206, 206])
y.append([159, 162])
y.append([0, 0])
y.append([255, 256])
y.append([146, 147])
y.append([224, 232])
y.append([237, 238])
y.append([282, 283])
y.append([304, 304])
y.append([115, 115])
y.append([103, 103])
y.append([133, 134])
y.append([124, 126])
y.append([124, 126])
y.append([193, 194])
y.append([196, 196])
y.append([142, 149])
y.append([228, 228])
y.append([228, 228])
y.append([178, 178])
y.append([230, 233])
y.append([169, 169])
y.append([166, 169])
y.append([164, 168])
y.append([135, 136])
y.append([126, 135])
y.append([126, 135])

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(100):
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        model.train()
        train_loss = train(train_batch, model, optimizer)
        train_loop.set_postfix(train_loss=train_loss.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Epoch {epoch} train")
    average_train_loss = overall_train_loss / num_of_train_batches
    print(f"average train loss: {average_train_loss}")

    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss = test(test_batch, model)
        test_loop.set_postfix(test_loss=test_loss.item())
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Epoch {epoch} test")
    average_test_loss = overall_test_loss / num_of_test_batches
    print(f"average test loss: {average_test_loss}")
