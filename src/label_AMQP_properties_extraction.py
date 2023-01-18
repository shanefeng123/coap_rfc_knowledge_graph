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
        decoded_test[i] = [decoded_test[i].split("\n")[0].split("Context:")[1].strip(),
                           decoded_test[i].split("\n")[1].split("Sentence:")[1].strip()]

    for sample in decoded_test:
        test_generated.append(
            tokenizer.decode(generate(sample[0], sample[1], model, tokenizer)[0], skip_special_tokens=True))
    return test_loss, test_decoded_labels, test_generated


def generate(context, sentence, model, tokenizer):
    data = f"<|startoftext|>Context: {context}\nSentence: {sentence}\nProperties: "
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024,
                               pad_token_id=tokenizer.eos_token_id)
    return generated


def construct_context(pronoun_sentence, specification_sentences, k):
    pronoun_sentence_index = specification_sentences.index(pronoun_sentence)
    context_start_index = pronoun_sentence_index - k
    context_sentences = specification_sentences[context_start_index:pronoun_sentence_index + 1]
    return " ".join(context_sentences)

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
rule_condition_split = pickle.load(open('../data/amqp_sentence_condition_split', 'rb'))

contexts = []
# Construct context for the split sentences
for i in range(len(rule_condition_split)):
    sentence_split = rule_condition_split[i].split("\n")
    original_sentence = sampled_AMQP_rule_sentences[i]
    for j in range(len(sentence_split)):
        split = sentence_split[j]
        if "Not applicable" in split:
            context = construct_context(original_sentence, amqp_spec_sentences, 5)
            contexts.append((context, original_sentence, "Entity rule", i))
            break
        else:
            context = construct_context(original_sentence, amqp_spec_sentences, 5)
            if split.startswith("Antecedent"):
                contexts.append((context, split.split("Antecedent:")[1].strip(), "Antecedent rule", i))
            else:
                contexts.append((context, split.split("Consequent:")[1].strip(), "Consequent rule", i))

y = []
y.append("peer @ support at least the en-US locale = True;")
y.append("handle ﬁeld @ is set = False;")
y.append("delivery-count @ be set = False;")
y.append("next-outgoing-id @ be initialized to an arbitrary value = True; next-outgoing-id @ is incremented after each successive transfer according to RFC-1982 [RFC1982] serial number arithmetic = True;")
y.append("bare message @ exact encoding of sections be modified = False;")
y.append("resent delivery @ be sent = True; resume ﬂag @ set to true = True; delivery-tag @ set to 4 = True;")
y.append("ﬂow information @ is given by the receiver = True;")
y.append("sender's link-credit variable @ be set according to this formula = True;")
y.append("BEGIN SENT state @ in this state = True;")
y.append("endpoint @ send frames = True; endpoint @ receive frames = False;")
# 10
y.append("peer @ needs to satisfy the need to send trafﬁc to prevent idle timeouts = True; peer @ has nothing to send = True;")
y.append("peer @ send an empty frame = True;")
y.append("requested protocol version @ is supported = True;")
y.append("server @ send its own protocol header with the requested version to the socket = True; server @ proceed according to the protocol deﬁnition;")
y.append("target @ capable of fulﬁlling this guarantee = False; durable header @ set to true = True;")
y.append("target @ accept messages = False;")
y.append("source @ allows the rejected outcome = True; link @ be detached by the receiver with the same error = False;")
y.append("message @ be rejected with the precondition-failed error = True;")
y.append("value @ be of the same type as would be valid in a ﬁeld deﬁned with the following attributes = True;")
y.append("handle @ currently associated with an attached link = False;")
# 20
y.append("recipient @ respond by ending the session with an unattached-handle session error = True;")
y.append("remote peer @ respond gracefully within a threshold to this = True;")
y.append("peer @ close the TCP socket = True;")
y.append("peer @ received an oversized frame = True;")
y.append("peer @ close the connection with the framing-error error-code = True;")
y.append("endpoint @ supports more than one distribution-mode = True;")
y.append("distribution mode @ be set by the sending end of the link = True;")
y.append("address name @ include the link name = True; address name @ include the container-id of the remote container = True;")
y.append("links @ closed = False; links @ detached with an error = False;")
y.append("deliveries @ alive = True; updated state @ be applied = True;")
# 30
y.append("endpoint @ reattaching = True;")
y.append("unsettled map @ be null = True;")
y.append("sender’s link-credit @ falls below a threshold = True;")
y.append("flow state @ be sent to increase the sender’s link- credit back to the desired target amount = True;")
y.append("remote endpoint outgoing-window @ exceed = False;")
y.append("remote-outgoing-window @ reﬂects the maximum number of incoming transfers = True;")
y.append("receiver @ initiates the attach exchange = True; sender @ supports the desired mode = True;")
y.append("sender @ respect the receiver’s desired settlement mode = True;")
y.append("sender @ indicate an aborted attempt = True;")
y.append("receiver @ discard the message data that was transferred prior to the abort = True;")
# 40
y.append("version @ less than or equal to the requested version = False;")
y.append("server @ respond with the highest supported version = True;")
y.append("sending peer @ require its partner to authenticate with it = True;")
y.append("sending peer @ send a list of one element with its value as the SASL mechanism ANONYMOUS = True;")
y.append("implementations @ always assign the lowest available handle to handle-in-use session-error = True;")
y.append("endpoint @ make use of the ability to send an incomplete unsettled map = True;")
y.append("Deliveries @ only the target considers unsettled Deliveries in this category = True;")
y.append("Deliveries @ be ignored by the sender = True; Deliveries @ be considered settled by the receiver = True;")
y.append("messages @ insufﬁcient to consume the current link-credit = True;")
y.append("drain ﬂag @ indicates how the sender SHOULD behave = True;")
# 50
y.append("sender @ makes multiple requests for the same state = True; receiver @ can reply = False;")
y.append("receiver @ send only one ﬂow in return = True;")
y.append("implementations @ operationally necessary = True;")
y.append("implementations @ choose to run this pure TLS server on other ports = True;")
y.append("sender @ send a mixture of settled and unsettled deliveries to the receiver = True;")
y.append("session @ locally initiated = True;")
y.append("remote-channel ﬁeld of a begin frame @ be empty = True; remote-channel ﬁeld of a begin frame @ be set when announcing the endpoint created as a result of a remotely initiated session = True;")
y.append("transaction @ discharged = True;")
y.append("controller @ settle any outstanding unsettled deliveries in a timely fashion = True;")
y.append("sender @ sets the aborted ﬂag to true = True;")
# 60
y.append("sender @ set the more ﬂag to true = False;")
y.append("hostname @ be supplied in the hostname ﬁeld of the open frame during SASL = True; hostname @ be supplied in the hostname ﬁeld of the open frame during TLS negotiation = True;")
y.append("link @ copy = True;")
y.append("state @ be retained at the source to ensure compliance = True;")
y.append("peer @ attempt to attach a link using a handle value outside the range that its partner can handle = False;")
y.append("Delivery @ be identiﬁed by a delivery-tag chosen by the sending application = True;")
y.append("endpoint @ responds to a remotely initiated session = True;")
y.append("the remote-channel @ be set to the channel on which the remote session sent the begin = True;")
y.append("transfer @ resuming transfers = True;")
y.append("sender @ set the delivery-state at the source = True;")
# 70
y.append("receiving endpoint @ sent an incomplete unsettled map = True; receiving endpoint @ receiving a transfer which does not have the resume ﬂag set to true = True;")
y.append("receiving endpoint @ detach with an error = True;")
y.append("endpoint @ receiving the remote endpoint’s corresponding end frame = False;")
y.append("endpoint @ proceed to discard all incoming frames from the remote endpoint = True;")
y.append("distribution modes @ be one or more symbols which are valid distribution-modes = True;")
y.append("handle ﬁeld @ is set = False;")
y.append("delivery-count @ be set = False;")
y.append("maps @ order known to otherwise = True;")
y.append("maps @ be considered to be ordered = True;")
y.append("handle ﬁeld @ is set = False;")
# 80
y.append("delivery-count @ be set = False;")
y.append("frames @ from resource to controller = True;")
y.append("transfer and disposition frames @ indicate work performed = True; txn-ids @ correctly indicate with which (if any) transaction this work is associated = True;")
y.append("peer @ responds to echo requests with ﬂows which themselves have the echo ﬁeld set to true = True; partner @ adopts the same policy = True;")
y.append("inﬁnite loop @ result = True;")
y.append("address of the source @ sent on a attach frame sent by the receiving link endpoint where the dynamic ﬂag is set to true (that is where the receiver is requesting the sender to create an addressable node) = True;")
y.append("address of the source @ be set = False;")
y.append("message @ is transmitted by an intermediary that was received with a ttl = True;")
y.append("transmitted message’s header @ contain a ttl that is computed as the difference between the current time and the formerly computed message expiration time = True;")
y.append("annotations @ are explicitly augmented or modiﬁed = False;")
# 90
y.append("Intermediaries @ propagate the annotations = True;")
y.append("transfer @ from controller to resource was associated = True;")
y.append("outcome communicated by the resource @ be associated with the same transaction = True;")
y.append("peer @ has received the begin frame for the session = True;")
y.append("next-incoming-id @ be set = True;")
y.append("peer @ has received the begin frame for the session = False;")
y.append("next-outgoing-id @ be set = False;")
y.append("content type @ unknown = True;")
y.append("content-type @ be set = False;")
y.append("delivery @ ever spontaneously attain the rejected state at the source = False;")
# 100
y.append("sender @ transfer messages even if the available variable is zero = True;")
y.append("senders @ be aware = True; implementations @ choose to use an internal default to efﬁciently manage a peer’s resources = True;")
y.append("echo request state from partner @ set to true = True;")
y.append("receiver @ send its state at the earliest convenient opportunity = True;")
y.append("handle @ already associated with a link = True;")
y.append("attempt @ be responded to with an immediate close carrying a handle-in-use session-error = True;")
y.append("peer @ receiving the partner’s connection header or open frame = False;")
y.append("peer @ do this by starting to send subsequent frames = True;")
y.append("state @ DISCARDING = True; peer @ close frame is received = False;")
y.append("peer @ any incoming frames on the connection be silently discarded = True;")
# 110
y.append("peer @ closing a connection = False;")
y.append("peer @ write a close frame with a code indicating the reason for closing = True;")
y.append("delivery-failed ﬂag @ is set = True;")
y.append("messages modiﬁed @ have their delivery-count incremented = True;")
y.append("implementations @ run a pure TLS server = True;")
y.append("dynamic ﬁeld @ set to true = False;")
y.append("dynamic-node-properties @ be set = False;")
y.append("address of the source @ sent on a attach frame sent by the receiving link endpoint where the dynamic ﬂag is set to true (that is where the receiver has created an addressable node at the request of the sender and is now communicating the address of that created node) = True;")
y.append("address of the source @ be set = True;")
y.append("hostname @ provided = False;")
y.append("receiving peer @ select a default based on its own conﬁguration = True;")




print(len(y))
data = []
for i in range(len(y)):
    data.append(
        f"<|startoftext|>Context: {contexts[i][0]}\nSentence: {contexts[i][1]}\nProperties: {y[i]}<|endoftext|>")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model/properties_extractor.pt", map_location=device)

inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
labels = inputs["input_ids"].clone()
inputs["labels"] = labels

dataset = MeditationDataset(inputs)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

test_loop = tqdm(test_loader, leave=True)
overall_test_loss = 0
num_of_test_batches = len(test_loader)
overall_test_bleu = 0
for test_batch in test_loop:
    test_batch_bleu = 0
    model.eval()
    test_loss, test_decoded_labels, test_generated = test(test_batch, model)
    test_loop.set_postfix(test_loss=test_loss.item())
    overall_test_loss += test_loss.item()

    for i in range(len(test_decoded_labels)):
        test_decoded_labels[i] = test_decoded_labels[i].split("Properties:", 1)[1].strip()
        test_generated[i] = test_generated[i].split("Properties:", 1)[1].strip()
        print(test_decoded_labels[i])
        print(test_generated[i])

    for i in range(len(test_decoded_labels)):
        test_batch_bleu += sentence_bleu([test_decoded_labels[i]], test_generated[i])
    overall_test_bleu += test_batch_bleu

average_test_loss = overall_test_loss / num_of_test_batches
print(f"average test loss: {average_test_loss}")
average_test_bleu = overall_test_bleu / num_of_test_batches
print(f"average test bleu: {average_test_bleu}")

with open(r"../results/AMQP_properties_extraction_benchmark.txt", "a") as file:
    file.write(f"test loss: {average_test_loss} test bleu: {average_test_bleu}\n")
    file.write("\n")
