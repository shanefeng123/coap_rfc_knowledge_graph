from transformers import BertTokenizer, BertForPreTraining, AdamW
from accelerate import Accelerator
from tqdm import tqdm
import random
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")

data = open("./data/pretrain_sentences.txt").read().split("\n")
data_size = len(data)

# Prepare the data for next sentence prediction
sentences_a = []
sentences_b = []
labels = []

# Balance out the data so we have 50% of true next sentences and 50% false next sentences
for i in range(data_size - 1):
    sentences_a.append(data[i])
    if random.random() > 0.5:
        if not data[i + 1].startswith("introduction"):
            sentences_b.append(data[i + 1])
            labels.append(1)
        else:
            sentences_b.append(data[random.randint(0, data_size - 1)])
            labels.append(0)
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


dataset = MeditationDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda")
# accelerator = Accelerator()
# device = accelerator.device
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)

# model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for epoch in range(500):
    loop = tqdm(loader, leave=True)
    overall_loss = 0
    epoch_mlm_predictions = None
    epoch_nsp_predictions = None
    epoch_mlm_labels = None
    epoch_nsp_labels = None
    num_of_batches = 0
    for batch in loop:
        num_of_batches += 1
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        next_sentence_labels = batch["next_sentence_labels"].to(device)
        token_labels = batch["token_labels"].to(device)
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        next_sentence_label=next_sentence_labels, labels=token_labels)
        loss = outputs.loss
        loss.backward()
        # accelerator.backward(loss)
        optimizer.step()
        loop.set_description(f"Epoch {epoch}")
        # Calculate mlm accuracy
        mlm_predictions = torch.argmax(outputs.prediction_logits, dim=-1)
        mlm_accuracy = torch.sum(torch.eq(mlm_predictions, token_labels)) / (
                token_labels.shape[0] * token_labels.shape[1])
        # Calculate nsp accuracy
        nsp_predictions = torch.argmax(outputs.seq_relationship_logits, dim=-1)
        nsp_accuracy = torch.sum(torch.eq(nsp_predictions, next_sentence_labels)) / (
            next_sentence_labels.shape[0])
        loop.set_postfix(loss=loss.item(), mlm_accuracy=mlm_accuracy.item(), nsp_accuracy=nsp_accuracy.item())
        # Store the epoch info for overall loss and accuracy calculation
        overall_loss += loss.item()
        if epoch_mlm_predictions is None:
            epoch_mlm_predictions = mlm_predictions
            epoch_nsp_predictions = nsp_predictions
            epoch_mlm_labels = token_labels
            epoch_nsp_labels = next_sentence_labels
        else:
            epoch_mlm_predictions = torch.cat((epoch_mlm_predictions, mlm_predictions), dim=0)
            epoch_nsp_predictions = torch.cat((epoch_nsp_predictions, nsp_predictions), dim=0)
            epoch_mlm_labels = torch.cat((epoch_mlm_labels, token_labels), dim=0)
            epoch_nsp_labels = torch.cat((epoch_nsp_labels, next_sentence_labels), dim=0)

    average_loss = overall_loss / num_of_batches
    epoch_mlm_accuracy = torch.sum(torch.eq(epoch_mlm_predictions, epoch_mlm_labels)) / (
            epoch_mlm_labels.shape[0] * epoch_mlm_labels.shape[1])
    epoch_nsp_accuracy = torch.sum(torch.eq(epoch_nsp_predictions, epoch_nsp_labels)) / (
        epoch_nsp_labels.shape[0])
    print(f"average loss: {average_loss}")
    print(f"epoch mlm accuracy: {epoch_mlm_accuracy.item()}")
    print(f"epoch nsp accuracy: {epoch_nsp_accuracy.item()}")

    with open(r"pretrain_results.txt", "a") as file:
        file.write(
            f"Epoch {epoch} average_loss: {average_loss} mlm_accuracy: {epoch_mlm_accuracy.item()} nsp_accuracy: {epoch_nsp_accuracy.item()}")
        file.write("\n")
    if epoch_mlm_accuracy.item() > 0.99 and epoch_nsp_accuracy.item() > 0.99:
        break

torch.save(model, "coap_BERT.pt")
