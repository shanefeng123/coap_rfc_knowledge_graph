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

mqtt_rule_sentences = []
for sentence in mqtt_spec_sentences:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            mqtt_rule_sentences.append(sentence)
            break

# Start from here, sample 20% mqtt rule sentences
random.seed(4)
sampled_mqtt_rule_sentences = random.sample(mqtt_rule_sentences, 63)
rule_condition_split = pickle.load(open('../data/mqtt_sentence_condition_split', 'rb'))

contexts = []
# Construct context for the split sentences
for i in range(len(rule_condition_split)):
    sentence_split = rule_condition_split[i].split("\n")
    original_sentence = sampled_mqtt_rule_sentences[i]
    for j in range(len(sentence_split)):
        split = sentence_split[j]
        if "Not applicable" in split:
            context = construct_context(original_sentence, mqtt_spec_sentences, 5)
            contexts.append((context, original_sentence, "Entity rule", i))
            break
        else:
            context = construct_context(original_sentence, mqtt_spec_sentences, 5)
            if split.startswith("Antecedent"):
                contexts.append((context, split.split("Antecedent:")[1].strip(), "Antecedent rule", i))
            else:
                contexts.append((context, split.split("Consequent:")[1].strip(), "Consequent rule", i))

y = []
y.append("Payload @ contains zero bytes = True;")
y.append(
    "Server @ process normally = True; retrained message @ be removed = True; subscribers @ receive a retained message = False;")
y.append("Client @ specified a Subscription Identifier for any of the overlapping subscriptions = True;")
y.append(
    "Server @ send those Subscription Identifiers in the message which is published as the result of the subscriptions = True;")
y.append("Client @ sets an Authentication Method in the CONNECT = True; Client @ received a CONNACK packet = False;")
y.append("Client @ send any packets other than AUTH or DISCONNECT packets = False;")
y.append("Server @ receives UNSUBSCRIBE = True;")
y.append("Server @ add any new messages which match the Topic Filters = False;")
y.append("PUBREL packet @ contain the same Packet Identifier as the original PUBLISH packet = True;")
y.append("Server @ rejects CONNECT = True; Server @ rejects AUTH packet = False;")
# 10
y.append("Server @ process any data sent by the Client after the CONNECT packet = False;")
y.append("Client @ greater than Topic Alias Maximum = True;")
y.append("Server @ send a Topic Alias in a PUBLISH packet to the Client = False;")
y.append("Will Flag @ is set to 1 = True; Will Retain @ is set to 1 = True;")
y.append("Server @ publish the Will Message as a retained message = True;")
y.append("flag bit @ is marked as 'Reserved' = True;")
y.append("flag bit @ reserved for future use = True; flag bit @ be set to the value listed = True;")
y.append("Server @ respond to an UNSUBSCRIBE request by sending an UNSUBACK packet = True;")
y.append(
    "Server @ in the process of sending a QoS 1 message to its chosen subscribing Client = True; connection @ breaks = True; Server @ has received an acknowledgement from the Client = False;")
y.append("Server @ wait for the Client to reconnect = True; Server @ retransmit the message to that Client = True;")
# 20
y.append("Server @ forwarding the Application Message to a Client = True;")
y.append("Server @ send all User Properties unaltered in a PUBLISH packet = True;")
y.append("Server @ shutdown = True; Server @ failure = True;")
y.append("Server @ defer publication of Will Messages until a subsequent restart = True;")
y.append("Server @ send the Server Keep Alive = False;")
y.append("Server @ use the Keep Alive value set by the Client on CONNECT = True;")
y.append("multi-level wildcard character @ be the last character specified in the Topic Filter = True;")
y.append(
    "Server @ use a security component to authorize particular actions on the topic resource for a given Client = True;")
y.append("Bit 2 of the Subscription Options @ value is 1 = True;")
y.append(
    "Application Messages @ be forwarded to a connection with a ClientID equal to the ClientID of the publishing connection = False;")
# 30
y.append("Topic Alias @ values greater than 0 = True; Topic Alias @ less than or equal to the Topic Alias Maximum value = True;")
y.append("Client @ accept all Topic Alias that it sent in the CONNECT packet = True;")
y.append("Client @ have Session State = True; Client @ receives Session Present set to 0 = True; Client @ continues with the Network Connection = True;")
y.append("Client @ discard its Session State = True;")
y.append("ClientID @ be used by Clients = True; ClientID @ be used by Servers = True;")
y.append("subscribers @ receiving the Application Message = True;")
y.append("Server @ send the Payload Format Indicator unaltered to all subscribers = True;")
y.append("Server @ increase the size of the CONNACK packet beyond the Maximum Packet Size specified by the Client = True;")
y.append("Server @ send this property = False;")
y.append("Server @ sends a new PUBLISH (with QoS > 0) MQTT Control Packet = True;")
# 40
y.append("Server @ assign it a non zero Packet Identifier that is currently unused = True;")
y.append("UTF-8 data @ in the Payload = True;")
y.append("UTF-8 data @ be well-formed UTF-8 as defined by the Unicode specification [Unicode] and restated in RFC 3629 [RFC3629] = True;")
y.append("Topic Alias @ has the value 0 = True;")
y.append("sender @ send a PUBLISH packet containing a Topic Alias = False;")
y.append("Server @ send CONNACK with Reason Code 0x9A (Retain not supported) = True; Server @ close the Network Connection = True;")
y.append("Server @ accepts a connection with Clean Start set to 1 = True;")
y.append("Server @ set Session Present to 0 in the CONNACK packet = True; Server @ setting a 0x00 (Success) Reason Code in the CONNACK packet = True;")
y.append("PUBLISH packet @ sent from a Client to a Server = True;")
y.append("PUBLISH packet @ contain a Subscription Identifier = True;")
# 50
y.append("SUBACK packet @ sent by the Server to the Client = True;")
y.append("SUBACK packet @ contain a Reason Code for each Topic Filter/Subscription Option pair = True;")
y.append("packets @ exceeding Maximum Packet Size = True;")
y.append("Server @ send packets to the Client = False;")
y.append("packet @ error = True; Server @ close the Network Connection = False;")
y.append("Server @ send a DISCONNECT packet containing the Reason Code = True;")
y.append("Server @ treat any other value as malformed = True; Server @ close the Network Connection = True;")
y.append("Server @ treat any other value as malformed = True; Server @ close the Network Connection = True;")
y.append("Client @ acknowledge any Publish packet it receives according to the applicable QoS rules = True;")
y.append("Retain As Published subscription option @ value is set to 0 = True; Server @ forwarding an Application Message = True;")
# 60
y.append("Server @ set the RETAIN flag to 0 = True;")
y.append("Server @ sends a CONNACK packet containing a Reason code of 128 or greater = True;")
y.append("Server @ close the Network Connection = True;")
y.append("Server @ receives a PUBLISH packet with the RETAIN flag set to 1, and QoS 0 = True;")
y.append("Server @ store the new QoS 0 message as the new retained message for that topic = True; Server @ choose to discard it at any time = True;")
y.append("sender @ send a PUBLISH packet containing this Packet Identifier with QoS 2 and DUP flag set to 0 = True;")
y.append("Topic Alias @ values greater than 0 = True; Topic Alias @ less than or equal to the Topic Alias Maximum value = True;")
y.append("Server @ accept all Topic Alias that it returned in the CONNACK packet = True;")
y.append("Packet @ is too large to send = True;")
y.append("Server @ discard it without sending it = True; Server @ behave as if it had completed sending that Application Message = True;")
# 70
y.append("Client @ Session terminates = True; Client @ reconnects = False;")
y.append("Server @ send the Application Message to any other subscribed Client = False;")
y.append("published message @ match multiple filters = True;")
y.append("Server @ deliver the message to the Client respecting the maximum QoS of all the matching subscriptions = True;")
y.append("UTF-8 Encoded String @ include an encoding of the null character U+0000 = False;")
y.append("Server @ forwarding the Application Message = True;")
y.append("Server @ maintain the order of User Properties = True;")
y.append("Authentication Method @ specifies that the Client sends data first = True;")
y.append("Client @ include an Authentication Data property in the CONNECT packet = True;")
y.append("Client @ received PUBACK, PUBCOMP, or PUBREC with a Reason Code of 128 or greater from the Server = False;")
# 80
y.append("Client @ send more than Receive Maximum QoS 1 and QoS 2 PUBLISH packets = False")
y.append("")
y.append("Reason Codes @ order in UNSUBACK packet = True;")
y.append("Reason Codes @ match the order of Topic Filters in the UNSUBSCRIBE packet = True;")
y.append("Client @ sending the DISCONNECT packet = True; Server @ sending the DISCONNECT packet = True;")
y.append("Client @ use one of the DISCONNECT Reason Code values = True; Server @ use one of the DISCONNECT Reason Code values = True;")
y.append("AUTH packet @ error = True; Client @ close the Network Connection = False;")
y.append("Client @ send a DISCONNECT packet containing the reason code = True;")
y.append("subscribers @ receiving the Application Message = True;")
y.append("Server @ send the Correlation Data unaltered to all subscribers = True;")
# 90
y.append("sender @ increase the size of the DISCONNECT packet beyond the Maximum Packet Size specified by the receiver = True;")
y.append("sender @ send this property = False;")
y.append("Network Connection @ is open = True;")
y.append("Client @ discard the Session State = False; Server @ discard the Session State = False;")
y.append("Client @ receive a CONNACK packet from the Server within a reasonable amount of time = False;")
y.append("Client @ close the Network Connection = True;")
y.append("RETAIN flag @ is set to 1 in a PUBLISH packet sent by a Client to a Server = True;")
y.append("Server @ replace any existing retained message for this topic = True; Server @ store the Application Message = True;")
y.append("Server @ sends a single copy of the message = True;")
y.append("Server @ include in the PUBLISH packet the Subscription Identifiers for all matching subscriptions which have a Subscription Identifiers = True;")
# 100
y.append("Topic Name @ in a PUBLISH packet sent by a Server to a subscribing Client = True;")
y.append("Topic Name @ match the Subscription’s Topic Filter according to the matching process = True;")
y.append("CONNECT packet @ is received with Clean Start is set to 1 = True;")
y.append("Client @ discard any existing Session and start a new Session = True; Server @ discard any existing Session and start a new Session = True;")
y.append("Keep Alive @ value is non-zero = True; Server @ receive an MQTT Control Packet from the Client within one and a half times the Keep Alive time period = False;")
y.append("Server @ close the Network Connection to the Client = True;")
y.append("CONNECT packet @ is received with Clean Start set to 0 = True; Session @ associated with the Client Identifier = True;")
y.append("Server @ resume communications with the Client based on state from the existing Session = True;")
y.append("QoS 1 delivery protocol @ is used = True;")
y.append("sender @ assign an unused Packet Identifier each time it has a new Application Message to publish = True;")
# 110
y.append("multi-level wildcard character @ be specified on its own = True; multi-level wildcard character @ following a topic level separator = True;")
y.append("Server @ receives the first subscription = True; Server @ receives an Application Message with that Topic Name = True;")
y.append("topic resource @ be predefined in the Server by an administrator = True; topic resource @ be dynamically created by the Server = True;")


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

with open(r"../results/MQTT_properties_extraction_benchmark.txt", "a") as file:
    file.write(f"test loss: {average_test_loss} test bleu: {average_test_bleu}\n")
    file.write("\n")
