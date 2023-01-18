from transformers import BertTokenizer, BertForPreTraining, AdamW
from accelerate import Accelerator
from tqdm import tqdm
import random
import torch
"""
This is following the method described in the video.
https://www.youtube.com/watch?v=IC9FaVPKlYc&t=1079s
"""
random.seed(4)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForPreTraining.from_pretrained("bert-base-cased")

data = open("../data/pretrain_sentences.txt").read().split("\n")
data_size = len(data)

# Prepare the data for next sentence prediction
sentences_a = []
sentences_b = []
labels = []

# Balance out the data, so we have 50% of true next sentences and 50% false next sentences
for i in range(data_size - 1):
    sentences_a.append(data[i])
    if random.random() > 0.5:
        sentences_b.append(data[i + 1])
        labels.append(1)

    else:
        sentences_b.append(data[random.randint(0, data_size - 1)])
        labels.append(0)

# Tokenize the input, create next sentence label tensor, mask 15% of the tokens for mask language modeling
inputs = tokenizer(sentences_a, sentences_b, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
inputs["next_sentence_labels"] = torch.LongTensor(labels).T

inputs["token_labels"] = inputs["input_ids"].detach().clone()
randomness = torch.rand(inputs["input_ids"].shape)
mask = (randomness < 0.15) * (inputs["input_ids"] != 101) * (inputs["input_ids"] != 102) * (inputs["input_ids"] != 0)

for i in range(inputs["input_ids"].shape[0]):
    selections = torch.flatten(mask[i].nonzero()).tolist()
    inputs["input_ids"][i, selections] = 103


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
    next_sentence_labels = batch["next_sentence_labels"].to(device)
    token_labels = batch["token_labels"].to(device)
    train_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                          next_sentence_label=next_sentence_labels, labels=token_labels)
    train_loss = train_outputs.loss
    train_loss.backward()
    # accelerator.backward(loss)
    optimizer.step()
    mlm_predictions = torch.argmax(train_outputs.prediction_logits, dim=-1)
    nsp_predictions = torch.argmax(train_outputs.seq_relationship_logits, dim=-1)
    # Calculate mlm accuracy
    mlm_accuracy = torch.sum(torch.eq(mlm_predictions, token_labels)) / (
            token_labels.shape[0] * token_labels.shape[1])
    # Calculate nsp accuracy
    nsp_accuracy = torch.sum(torch.eq(nsp_predictions, next_sentence_labels)) / (
        next_sentence_labels.shape[0])
    return train_loss, mlm_predictions, nsp_predictions, mlm_accuracy, nsp_accuracy, token_labels, next_sentence_labels


def test(batch, model):
    model.eval()
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    next_sentence_labels = batch["next_sentence_labels"].to(device)
    token_labels = batch["token_labels"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         next_sentence_label=next_sentence_labels, labels=token_labels)
    test_loss = test_outputs.loss
    mlm_predictions = torch.argmax(test_outputs.prediction_logits, dim=-1)
    nsp_predictions = torch.argmax(test_outputs.seq_relationship_logits, dim=-1)
    # Calculate mlm accuracy
    mlm_accuracy = torch.sum(torch.eq(mlm_predictions, token_labels)) / (
            token_labels.shape[0] * token_labels.shape[1])
    # Calculate nsp accuracy
    nsp_accuracy = torch.sum(torch.eq(nsp_predictions, next_sentence_labels)) / (
        next_sentence_labels.shape[0])
    return test_loss, mlm_predictions, nsp_predictions, mlm_accuracy, nsp_accuracy, token_labels, next_sentence_labels


#
# prepare dataset and split it into train and test set
dataset = MeditationDataset(inputs)
dataset_length = len(dataset)
train_length = int(dataset_length * 0.8)
test_length = dataset_length - train_length
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
#
device = torch.device("cuda")
# accelerator = Accelerator()
# device = accelerator.device
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)

# model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for epoch in range(500):
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    epoch_train_mlm_predictions = None
    epoch_train_nsp_predictions = None
    epoch_train_mlm_labels = None
    epoch_train_nsp_labels = None
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        train_loss, train_mlm_predictions, train_nsp_predictions, train_mlm_accuracy, train_nsp_accuracy, train_mlm_labels, train_nsp_labels = train(
            train_batch,
            model,
            optimizer)

        train_loop.set_postfix(train_loss=train_loss.item(), train_mlm_accuracy=train_mlm_accuracy.item(),
                               train_nsp_accuracy=train_nsp_accuracy.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Epoch {epoch} train")

        if epoch_train_mlm_predictions is None:
            epoch_train_mlm_predictions = train_mlm_predictions
            epoch_train_nsp_predictions = train_nsp_predictions
            epoch_train_mlm_labels = train_mlm_labels
            epoch_train_nsp_labels = train_nsp_labels
        else:
            epoch_train_mlm_predictions = torch.cat((epoch_train_mlm_predictions, train_mlm_predictions), dim=0)
            epoch_train_nsp_predictions = torch.cat((epoch_train_nsp_predictions, train_nsp_predictions), dim=0)
            epoch_train_mlm_labels = torch.cat((epoch_train_mlm_labels, train_mlm_labels), dim=0)
            epoch_train_nsp_labels = torch.cat((epoch_train_nsp_labels, train_nsp_labels), dim=0)

    # Evaluate on test data
    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    epoch_test_mlm_predictions = None
    epoch_test_nsp_predictions = None
    epoch_test_mlm_labels = None
    epoch_test_nsp_labels = None
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss, test_mlm_predictions, test_nsp_predictions, test_mlm_accuracy, test_nsp_accuracy, test_mlm_labels, test_nsp_labels = test(
            test_batch, model)
        test_loop.set_postfix(test_loss=test_loss.item(), test_mlm_accuracy=test_mlm_accuracy.item(),
                              test_nsp_accuracy=test_nsp_accuracy.item())
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Epoch {epoch} test")
        if epoch_test_mlm_predictions is None:
            epoch_test_mlm_predictions = test_mlm_predictions
            epoch_test_nsp_predictions = test_nsp_predictions
            epoch_test_mlm_labels = test_mlm_labels
            epoch_test_nsp_labels = test_nsp_labels
        else:
            epoch_test_mlm_predictions = torch.cat((epoch_test_mlm_predictions, test_mlm_predictions), dim=0)
            epoch_test_nsp_predictions = torch.cat((epoch_test_nsp_predictions, test_nsp_predictions), dim=0)
            epoch_test_mlm_labels = torch.cat((epoch_test_mlm_labels, test_mlm_labels), dim=0)
            epoch_test_nsp_labels = torch.cat((epoch_test_nsp_labels, test_nsp_labels), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    epoch_train_mlm_accuracy = torch.sum(torch.eq(epoch_train_mlm_predictions, epoch_train_mlm_labels)) / (
            epoch_train_mlm_labels.shape[0] * epoch_train_mlm_labels.shape[1])
    epoch_train_nsp_accuracy = torch.sum(torch.eq(epoch_train_nsp_predictions, epoch_train_nsp_labels)) / (
        epoch_train_nsp_labels.shape[0])

    average_test_loss = overall_test_loss / num_of_test_batches
    epoch_test_mlm_accuracy = torch.sum(torch.eq(epoch_test_mlm_predictions, epoch_test_mlm_labels)) / (
            epoch_test_mlm_labels.shape[0] * epoch_test_mlm_labels.shape[1])
    epoch_test_nsp_accuracy = torch.sum(torch.eq(epoch_test_nsp_predictions, epoch_test_nsp_labels)) / (
        epoch_test_nsp_labels.shape[0])
    print(f"average train loss: {average_train_loss}")
    print(f"epoch train mlm accuracy: {epoch_train_mlm_accuracy.item()}")
    print(f"epoch train nsp accuracy: {epoch_train_nsp_accuracy.item()}")
    print(f"average test loss: {average_test_loss}")
    print(f"epoch test mlm accuracy: {epoch_test_mlm_accuracy.item()}")
    print(f"epoch test nsp accuracy: {epoch_test_nsp_accuracy.item()}")

    with open(r"../results/pretrain_results.txt", "a") as file:
        file.write(
            f"Epoch {epoch} average_train_loss: {average_train_loss} mlm_train_accuracy: {epoch_train_mlm_accuracy.item()} nsp_train_accuracy: {epoch_train_nsp_accuracy.item()}")
        file.write("\n")
        file.write(
            f"Epoch {epoch} average_test_loss: {average_test_loss} mlm_test_accuracy: {epoch_test_mlm_accuracy.item()} nsp_test_accuracy: {epoch_test_nsp_accuracy.item()}")
        file.write("\n")
    if epoch_train_mlm_accuracy.item() > 0.99 and epoch_test_mlm_accuracy.item() > 0.99:
        break

# torch.save(model, "coap_BERT.pt")
model.save_pretrained("../model/iot_bert")
