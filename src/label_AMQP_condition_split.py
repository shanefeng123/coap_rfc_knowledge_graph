import random
import pdfplumber
import nltk
import re
import torch
from transformers import GPT2Tokenizer
import pickle
from sklearn import metrics
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
LABELS = ["B-entity", "I-entity", "Other", "PAD"]
all_entity_indexes = []

class MeditationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    test_loss = test_outputs.loss
    test_decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_test = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    test_generated = []
    for i in range(len(decoded_test)):
        decoded_test[i] = decoded_test[i].split("\n")[0].split("Sentence:")[1].strip()

    for sentence in decoded_test:
        test_generated.append(tokenizer.decode(generate(sentence, model, tokenizer)[0], skip_special_tokens=True))
    return test_loss, test_decoded_labels, test_generated


def generate(sentence, model, tokenizer):
    data = f"<|startoftext|>Sentence: {sentence}\nAntecedent:"
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024,
                               pad_token_id=tokenizer.eos_token_id)
    return generated

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

random.seed(4)
sampled_AMQP_rule_sentences = random.sample(amqp_rule_sentences, 67)

# Label the condition split
y = []
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: When the handle ﬁeld is not set\nConsequent: this ﬁeld MUST NOT be set;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: when ﬂow information is given by the receiver;\nConsequent: This means that the sender’s link-credit variable MUST be set according to this formula:")
y.append("Antecedent: In this state;\nConsequent: the endpoint MAY send frames but cannot receive them;")
y.append("Antecedent: If a peer needs to satisfy the need to send trafﬁc to prevent idle timeout, and has nothing to send;\nConsequent: it MAY send an empty frame, i.e., a frame consisting solely of a frame header, with no frame body;")
y.append("Antecedent: If the requested protocol version is supported;\nConsequent: the server MUST send its own protocol header with the requested version to the socket, and then proceed according to the protocol deﬁnition;")
y.append("Antecedent: A target which is not capable of fulﬁlling this guarantee, where the durable header is set to true;\nConsequent: MUST NOT accept messages;\nAntecedent: if the source allows the rejected outcome, otherwise the link MUST be detached by the receiver with the same error;\nConsequent: the message SHOULD be rejected with the precondition-failed error;")
# 10
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: If set to a handle that is not currently associated with an attached link;\nConsequent: the recipient MUST respond by ending the session with an unattached-handle session error;")
y.append("Antecedent: If the remote peer does not respond gracefully within a threshold to this;\nConsequent: then the peer MAY close the TCP socket;")
y.append("Antecedent: A peer that receives an oversized frame;\nConsequent: MUST close the connection with the framing-error error-code;")
y.append("Antecedent: if the endpoint supports more than one distribution-mode;\nConsequent: This ﬁeld MUST be set by the sending end of the link;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: As long as the links have not been closed or detached with an error;\nConsequent: then the deliveries are still “live” and the updated state MUST be applied;")
y.append("Antecedent: When reattaching (as opposed to resuming);\nConsequent: the unsettled map MUST be null;")
y.append("Antecedent: When the sender’s link-credit falls below a threshold;\nConsequent: the flow state MAY be sent to increase the sender’s link- credit back to the desired target amount;")
y.append("Antecedent: without exceeding the remote endpoint’s outgoing-window;\nConsequent: The remote-outgoing-window reﬂects the maximum number of incoming transfers;")
# 20
y.append("Antecedent: if the receiver initiates the attach exchange and the sender supports the desired mode;\nConsequent: The sender SHOULD respect the receiver’s desired settlement mode;")
y.append("Antecedent: In this case;\nConsequent: the receiver MUST discard the message data that was transferred prior to the abort;")
y.append("Antecedent: If no such version exists;\nConsequent: the server SHOULD respond with the highest supported version;")
y.append("Antecedent: If the sending peer does not require its partner to authenticate with it;\nConsequent: then it SHOULD send a list of one element with its value as the SASL mechanism ANONYMOUS;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Deliveries that only the target considers unsettled Deliveries in this category;\nConsequent: MUST be ignored by the sender, and MUST be considered settled by the receiver;")
y.append("Antecedent: when insufﬁcient messages are available to consume the current link-credit;\nConsequent: The drain ﬂag indicates how the sender SHOULD behave;")
y.append("Antecedent: If a sender makes multiple requests for the same state before the receiver can reply;\nConsequent: the receiver MAY send only one ﬂow in return;")
y.append("Antecedent: if this is operationally necessary;\nConsequent: Imple- mentations MAY also choose to run this pure TLS server on other ports;")
# 30
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: for a locally initiated session; \nConsequent: The remote-channel ﬁeld of a begin frame MUST be empty, and MUST be set when announcing the endpoint created as a result of a remotely initiated session;")
y.append("Antecedent: after the transaction has discharged;\nConsequent: The controller SHOULD settle any outstanding unsettled deliveries in a timely fashion;")
y.append("Antecedent: if it also sets the aborted ﬂag to true;\nConsequent: A sender SHOULD NOT set the more ﬂag to true;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: For a copy link;\nConsequent: state MUST be retained at the source to ensure compliance;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: When an endpoint responds to a remotely initiated session;\nConsequent: the remote-channel MUST be set to the channel on which the remote session sent the begin;")
y.append("Antecedent: On this transfer;\nConsequent: the sender SHOULD set the delivery-state at the source;")
# 40
y.append("Antecedent: a receiving endpoint which sent an incomplete unsettled map, receiving a transfer which does not have the resume ﬂag set to true;\nConsequent: MUST detach with an error;")
y.append("Antecedent: until receiving the remote endpoint’s corresponding end frame;\nConsequent: It MUST then proceed to discard all incoming frames from the remote endpoint;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: When the handle ﬁeld is not set;\nConsequent: this ﬁeld MUST NOT be set;")
y.append("Antecedent: Unless known to be otherwise;\nConsequent: maps MUST be considered to be ordered, that is, the order of the key-value pairs is semantically important and two maps which are different only in the order in which their key-value pairs are encoded are not equal;")
y.append("Antecedent: When the handle ﬁeld is not set;\nConsequent: this ﬁeld MUST NOT be set;")
y.append("Antecedent: When traveling in the other direction, from resource to controller;\nConsequent:  the transfer and disposition frames indicate work performed, and the txn-ids included MUST correctly indicate with which (if any) transaction this work is associated;")
y.append("Antecedent: if a peer responds to echo requests with ﬂows which themselves have the echo ﬁeld set to true, if its partner adopts the same policy (therefore such a policy SHOULD be avoided);\nConsequent: an inﬁnite loop could result;")
y.append("Antecedent: when sent on a attach frame sent by the receiving link endpoint where the dynamic ﬂag is set to true (that is where the receiver is requesting the sender to create an addressable node);\nConsequent: The address of the source MUST NOT be set;")
y.append("Antecedent: When a message is transmitted by an intermediary that was received with a ttl;\nConsequent: the transmitted message’s header SHOULD contain a ttl that is computed as the difference between the current time and the formerly computed message expiration time, i.e., the reduced ttl, so that messages will eventually die if they end up in a delivery loop;")
# 50
y.append("Antecedent: unless the annotations are explicitly augmented or modiﬁed (e.g., by the use of the modified outcome);\nConsequent: Intermediaries MUST propagate the annotations;")
y.append("Antecedent: which the transfer from controller to resource was associated;\nConsequent: The outcome communicated by the resource MUST be associated with the same transaction;")
y.append("Antecedent: if the peer has received the begin frame for the session;\nConsequent: This value MUST be set;\nAntecedent: if it has not;\nConsequent: MUST NOT be set;")
y.append("Antecedent: where the content type is unknown;\nConsequent: the content-type SHOULD NOT be set;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: echo request state from partner If set to true;\nConsequent: then the receiver SHOULD send its state at the earliest convenient opportunity;")
y.append("Antecedent: An attempt to attach using a handle which is already associated with a link;\nConsequent: MUST be responded to with an immediate close carrying a handle-in-use session-error;")
y.append("Antecedent: before receiving the partner’s connection header or open frame;\nConsequent: A peer MAY do this by starting to send subsequent frames;")
# 60
y.append("Antecedent: In this case, until the peer’s close frame is received;\nConsequent: any incoming frames on the connection MUST be silently discarded;")
y.append("Antecedent: Prior to closing a connection;\nConsequent: each peer MUST write a close frame with a code indicating the reason for closing;")
y.append("Antecedent: If the delivery-failed ﬂag is set;\nConsequent: any messages modiﬁed MUST have their delivery-count incre- mented;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: If the dynamic ﬁeld is not set to true;\nConsequent: this ﬁeld MUST be left unset;")
y.append("Antecedent: when sent on a attach frame sent by the receiving link endpoint where the dynamic ﬂag is set to true (that is where the receiver has created an addressable node at the request of the sender and is now communicating the address of that created node);\nConsequent: The address of the source MUST be set;")
y.append("Antecedent: If no hostname is provided;\nConsequent: the receiving peer SHOULD select a default based on its own conﬁguration;")


with open("../data/amqp_sentence_condition_split", "wb") as file:
    pickle.dump(y, file)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model/condition_splitter.pt", map_location=device)

print(len(y))
data = []
for i in range(len(y)):
    data.append(f"<|startoftext|>Sentence: {sampled_AMQP_rule_sentences[i]}\n{y[i]}<|endoftext|>")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")

inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
labels = inputs["input_ids"].clone()
inputs["labels"] = labels
dataset = MeditationDataset(inputs)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

test_loop = tqdm(test_loader, leave=True)
overall_test_loss = 0
num_of_test_batches = len(test_loader)
overall_test_bleu = 0
for test_batch in test_loop:
    test_batch_bleu = 0
    model.eval()
    test_loss, test_decoded_labels, test_generated = test(test_batch, model)
    for i in range(len(test_decoded_labels)):
        test_decoded_labels[i] = test_decoded_labels[i].split("Antecedent:", 1)[1].strip()
        test_generated[i] = test_generated[i].split("Antecedent:", 1)[1].strip()
        print(test_decoded_labels[i])
        print(test_generated[i])
        test_batch_bleu += sentence_bleu([test_decoded_labels[i]], test_generated[i])
    test_loop.set_postfix(test_loss=test_loss.item())
    overall_test_loss += test_loss.item()
    overall_test_bleu += test_batch_bleu / len(test_decoded_labels)

average_test_loss = overall_test_loss / num_of_test_batches
print(f"average test loss: {average_test_loss}")
average_test_bleu = overall_test_bleu / num_of_test_batches
print(f"average test bleu: {average_test_bleu}")

with open("../data/amqp_sentence_condition_split", "wb") as file:
    pickle.dump(y, file)

with open(r"../results/AMQP_condition_split_benchmark.txt", "a") as file:
    file.write(f"test loss: {average_test_loss} test bleu: {average_test_bleu}\n")
    file.write("\n")