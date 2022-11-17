from transformers import PegasusForConditionalGeneration, PegasusTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from prepare_pretrain_data import prepare_pretrain_data
import re
from tqdm import tqdm
import pickle


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
    return train_loss


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    test_loss = test_outputs.loss
    decoded_test = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    for sample in decoded_test:
        context = sample.split("\n")[0].split("Context: ")[1]
        sentence = sample.split("\n")[1].split("Sentence: ")[1]
        print(tokenizer.decode(generate(context, sentence, model, tokenizer)[0]))
    return test_loss


def generate(context, sentence, model, tokenizer):
    data = f"<|startoftext|>Context: {context}\nSentence: {sentence}\nProperties: "
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024)
    return generated


def construct_context(pronoun_sentence, specification_sentences, k):
    pronoun_sentence_index = specification_sentences.index(pronoun_sentence)
    context_start_index = pronoun_sentence_index - k
    context_sentences = specification_sentences[context_start_index:pronoun_sentence_index + 1]
    return " ".join(context_sentences)


rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")
MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]

rfc7252_rule_sentences = []
for sentence in rfc7252:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rfc7252_rule_sentences.append(sentence)
            break
rfc7252_rule_sentences = rfc7252_rule_sentences[1:]
rule_condition_split = pickle.load(open('../data/sentence_condition_split', 'rb'))

contexts = []
# Construct context for the split sentences
for i in range(len(rule_condition_split)):
    sentence_split = rule_condition_split[i].split("\n")
    original_sentence = rfc7252_rule_sentences[i]
    for j in range(len(sentence_split)):
        split = sentence_split[j]
        if "Not applicable" in split:
            context = construct_context(original_sentence, rfc7252, 5)
            contexts.append((context, original_sentence, "Entity rule", i))
            break
        else:
            context = construct_context(original_sentence, rfc7252, 5)
            if split.startswith("Antecedent"):
                contexts.append((context, split.split(":")[1].strip(), "Antecedent rule", i))
            else:
                contexts.append((context, split.split(":")[1].strip(), "Consequent rule", i))

# The form of label is "Entity @ variable operator value;"
y = []
y.append("CoAP version number @ set to 1 = True;")
y.append("version number @ unknown = True;")
y.append("Messages @ be silently ignored = True;")
y.append("Token Length @ Lengths 9@15 = True;")
y.append("message @ be sent = True; Message @ be processed as a message format error = True;")
y.append("message @ presence of a marker followed by a zero@length payload = True;")
y.append("message @ be processed as a message format error = True;")
y.append(
    "Option Numbers @ instances appear in order of their Option Numbers = True; Option Numbers @ delta encoding is used = True;")
y.append("Option Delta @ set to 15 = True; Option Delta @ entire byte is the payload marker = False;")
y.append("message @ be processed as a message format error = True;")
# 10
y.append("Option Length @ set to 15 = True;")
y.append("message @ be processed as a message format error = True;")
y.append("Option Value @ length and format define variable@length values = True;")
y.append("Options @ defined in other documents = True;")
y.append("Options @ make use of other option value formats = True;")
y.append("sender @ has a choice = True;")
y.append("sender @ represent the integer with as few bytes as possible = True;")
y.append("recipient @ be prepared to process values with leading zero bytes = True;")
y.append("Token Length @ set to 0 = True; Message ID @ bytes of data be present after the Message ID field = False;")
y.append("Message ID @ bytes of data be present after the Message ID field = True;")
# 20
y.append("message @ be processed as a message format error = True;")
y.append("recipient @ lacks context to process the message properly = True;")
y.append(
    "recipient @ reject the message = True; recipient @ acknowledge a Confirmable message with an Acknowledgement message = True;")
y.append(
    "Acknowledgement message @ echo the Message ID of the Confirmable message = True; Acknowledgement message @ carry a response = True; Acknowledgement message @ be Empty = True;")
y.append("Reset message @ echo the Message ID of the Confirmable message = True; Reset message @ be Empty = True;")
y.append("recipient @ receive Acknowledgement messages = True; recipient @ receive Reset messages = True;")
y.append("recipient @ respond with either Acknowledgement or Reset messages = False;")
y.append("CoAP endpoint @ wait for acknowledgement = True; CoAP endpoint @ wait for reset = True;")
y.append("CoAP endpoint @ keep track of timeout; CoAP endpoint @ keep track of retransmission counter = True;")
y.append("Retransmission @ the entire sequence of (re@)transmissions stay in the envelope of MAX_TRANSMIT_SPAN = True;")
# 30
y.append("CoAP endpoint @ sent a Confirmable message = True;")
y.append("CoAP endpoint @ give up in attempting to obtain an ACK = True;")
y.append("responder @ needed = True;")
y.append(
    "responder @ rely on this cross@layer behavior from a requester = False; responder @ retain the state to create the ACK for the request = True;")
y.append("retransmission @ receipt of ICMP errors = True;")
y.append("retransmission @ give up retransmission = True;")
y.append("implementation @ take account of ICMP errors = True;")
y.append("implementation @ check the information about the original datagram in the ICMP message = True;")
y.append("implementation @ check the information about the original datagram in the ICMP message = False;")
y.append("implementation @ ICMP errors be ignored = True;")
# 40
y.append("implementation @ implementation note is followed = True;")
y.append("Packet Too Big errors @ be ignored = True;")
y.append("implementation @ implementation note is followed = False;")
y.append("Packet Too Big errors @ feed into a path MTU discovery algorithm = True;")
y.append("Source Quench ICMP messages @ be ignored = True; Time Exceeded ICMP messages @ be ignored = True;")
y.append(
    "Host error @ appropriate vetting = True; network error @ appropriate vetting = True; port error @ appropriate vetting = True; protocol unreachable errors @ appropriate vetting = True; parameter problem errors @ appropriate vetting = True;")
y.append(
    "Host error @ be used to inform the application of a failure in sending = True; network error @ be used to inform the application of a failure in sending = True; port error @ be used to inform the application of a failure in sending = True; protocol unreachable errors @ be used to inform the application of a failure in sending = True; parameter problem errors @ be used to inform the application of a failure in sending = True;")
y.append(
    "Non@confirmable message @ carries either a request or response = True; Non@confirmable message @ be Empty = False;")
y.append("Non@confirmable message @ be acknowledged by the recipient = False;")
y.append("recipient @ lacks context to process the message properly = True;")
# 50
y.append("recipient @ reject the message = True;")
y.append("recipient @ reject a Non@confirmable message = True;")
y.append("recipient @ send a matching Reset message = True; rejected message @ be ignored = True;")
y.append("sender @ transmit multiple copies of a Non@confirmable message within MAX_TRANSMIT_SPAN = True;")
y.append(
    "recipient @ echo Message ID in Acknowledgement message = True; recipient @ echo Message ID in Reset message = True;")
y.append("Message ID @ be reused within the EXCHANGE_LIFETIME = False;")
y.append(
    "Acknowledgment message @ match Confirmable message = True; Acknowledgment message @ match Non@confirmable message = True; Reset message @ match Confirmable message = True; Reset message @ match Non@confirmable message = True;")
y.append(
    "Acknowledgment message @ match Message ID of Confirmable message = True; Acknowledgement message @ match source endpoint with destination endpoint of Confirmable message = True; Acknowledgement message @ match Message ID of Non@confirmable message = True; Acknowledgement message @ match source endpoint with destination endpoint of Non@confirmable message = True; Reset message @ match Message ID of Confirmable message = True; Reset message @ match source endpoint with destination endpoint of Confirmable message = True; Reset message @ match Message ID of Non@confirmable message = True; Reset message @ match source endpoint with destination endpoint of Non@confirmable message = True;")
y.append(
    "recipient @ acknowledge each duplicate copy of a Confirmable message using the same Acknowledgement or Reset message = True; recipient @ process any request or response in the message only once = True;")
y.append(
    "Confirmable message @ transports a request that is idempotent = True; Confirmable message @ transports a request can be handled in an idempotent fashion = True;")
# 60
y.append(
    "recipient @ acknowledge each duplicate copy of a Confirmable message using the same Acknowledgement or Reset message = False; recipient @ process any request or response in the message only once = False;")
y.append("recipient @ ignore any duplicated Non-confirmable messages = True; recipient @ process any request or response in the message only once = True;")
y.append("CoAP message @ appropriately encapsulated = True;")
y.append("CoAP message @ fit within a single IP packet = True; CoAP message @ fit within a single IP datagram = True;")
y.append("Path MTU @ known for a destination = False;")
y.append("IP MTU @ 1280 bytes be assumed = True;")
y.append("headers @ nothing is known about the size = True;")
y.append("message @ 1152 bytes size upper bound = True; payload @ 1024 bytes size upper bound = True;")
y.append("clients @ limit the number of simultaneous outstanding interactions to NSTART = True;")
y.append("algorithm @ modified by additional congestion control optimizations = False;")
# 70
y.append("algorithm @ be chosen in such a way that an endpoint does not exceed an average data rate of PROBING_RATE = True;")
y.append("server @ ")



print(len(y))
data = []
for i in range(len(y)):
    data.append(
        f"<|startoftext|>Context: {contexts[i][0]}\nSentence: {contexts[i][1]}\nProperties: {y[i]}<|endoftext|>")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
labels = inputs["input_ids"].clone()
inputs["labels"] = labels

dataset = MeditationDataset(inputs)
dataset_length = len(dataset)
train_length = int(dataset_length * 0.8)
test_length = dataset_length - train_length
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

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

    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss = test(test_batch, model)
        test_loop.set_postfix(test_loss=test_loss.item())
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Epoch {epoch} test")

    average_train_loss = overall_train_loss / num_of_train_batches
    print(f"average train loss: {average_train_loss}")
    average_test_loss = overall_test_loss / num_of_test_batches
    print(f"average test loss: {average_test_loss}")

    if average_train_loss < 0.1:
        break
