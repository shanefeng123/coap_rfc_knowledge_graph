from prepare_pretrain_data import prepare_pretrain_data
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
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
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    train_loss = train_outputs.loss
    train_loss.backward()
    optimizer.step()
    # predictions = torch.argmax(train_outputs.logits, dim=-1)
    # accuracy = torch.sum(torch.eq(predictions, labels)) / (
    #         labels.shape[0] * labels.shape[1])
    return train_loss


def generate(batch, model):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
    return generated


rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]

rule_sentences = []
for sentence in rfc7252:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rule_sentences.append(sentence)
            break

rule_sentences = rule_sentences[1:]

data = []
# train_data = []
# test_data = []
# dataset_length = len(y)
# train_length = int(dataset_length * 0.8)
for i in range(len(rule_sentences)):
    data.append(f"<|startoftext|>Sentence: {rule_sentences[i]}\nBehaviours:")

# for i in range(train_length, dataset_length):
#     test_data.append(f"<|startoftext|>Sentence: {rule_sentences[i]}\nBehaviours:")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")

# inputs = tokenizer("", return_tensors="pt", padding=True, truncation=True)
# labels = inputs["input_ids"].clone()
# inputs["labels"] = labels

# inputs = tokenizer(data, return_tensors="pt", padding=False, truncation=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.load("../model/variable_extractor_GPT-2.pt", map_location=device)
model.to(device)

generated = []
for i in range(len(data)):
    tokenized = tokenizer(data[i], return_tensors="pt", padding=False, truncation=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=300, pad_token_id=tokenizer.eos_token_id,
        do_sample=False, temperature=0, top_k=0, top_p=0)
    generated.append(tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
    print(f"{i}: {generated[i]}")

for i in range(len(generated)):
    if not generated[i].endswith(";"):
        print(i)
