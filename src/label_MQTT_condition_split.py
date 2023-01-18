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

# Label the condition split
y = []
y.append(
    "Antecedent: If the Payload contains zero bytes;\nConsequent: it is processed normally by the Server but any retained message with the same topic name MUST be removed and any future subscribers for the topic will not receive a retained message;")
y.append(
    "Antecedent: If the Client specified a Subscription Identifier for any of the overlapping subscriptions;\nConsequent: the Server MUST send those Subscription Identifiers in the message which is published as the result of the subscriptions;")
y.append(
    "Antecedent: If a Client sets an Authentication Method in the CONNECT, until it has received a CONNACK packet;\nConsequent: the Client MUST NOT send any packets other than AUTH or DISCONNECT packets;")
y.append(
    "Antecedent: When a Server receives UNSUBSCRIBE;\nConsequent: It MUST stop adding any new messages which match the Topic Filters, for delivery to the Client;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the Server rejects the CONNECT, except AUTH packets;\nConsequent: it MUST NOT process any data sent by the Client after the CONNECT packet;")
y.append(
    "Antecedent: Client greater than Topic Alias Maximum;\nConsequent: The Server MUST NOT send a Topic Alias in a PUBLISH packet to the Client;")
y.append(
    "Antecedent: If the Will Flag is set to 1 and Will Retain is set to 1;\nConsequent: the Server MUST publish the Will Message as a retained message;")
y.append(
    "Antecedent: Where a flag bit is marked as 'Reserved';\nConsequent: it is reserved for future use and MUST be set to the value listed;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 10
y.append("Antecedent: If the Server is in the process of sending a QoS 1 message to its chosen subscribing Client and the connection to that Client breaks before the Server has received an acknowledgement from the Client;\nConsequent: the Server MAY wait for the Client to reconnect and retransmit the message to that Client;")
y.append("Antecedent: when forwarding the Application Message to a Client;\nConsequent: The Server MUST send all User Properties unaltered in a PUBLISH packet;")
y.append("Antecedent: In the case of a Server shutdown or failure;\nConsequent: the Server MAY defer publication of Will Messages until a subsequent restart;")
y.append("Antecedent: If the Server does not send the Server Keep Alive;\nConsequent: the Server MUST use the Keep Alive value set by the Client on CONNECT;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: If the value is 1;\nConsequent: Application Messages MUST NOT be forwarded to a connection with a ClientID equal to the ClientID of the publishing connection;")
y.append("Antecedent: Topic Alias values greater than 0 and less than or equal to the Topic Alias Maximum value;\nConsequent: A Client MUST accept all Topic Alias that it sent in the CONNECT packet;")
y.append("Antecedent: If the Client does have Session State and receives Session Present set to 0, if it continues with the Network Connection;\nConsequent: it MUST discard its Session State;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 20
y.append("Antecedent: subscribers receiving the Application Message;\nConsequent: A Server MUST send the Payload Format Indicator unaltered to all subscribers;")
y.append("Antecedent: if it would increase the size of the CONNACK packet beyond the Maximum Packet Size specified by the Client;\nConsequent: The Server MUST NOT send this property;")
y.append("Antecedent: Each time a Server sends a new PUBLISH (with QoS > 0) MQTT Control Packet;\nConsequent: it MUST assign it a non zero Packet Identifier that is currently unused;")
y.append("Antecedent: The UTF-8 data in the Payload;\nConsequent: MUST be well-formed UTF-8 as defined by the Unicode specification [Unicode] and restated in RFC 3629 [RFC3629];")
y.append("Antecedent: Topic Alias which has the value 0;\nConsequent: A sender MUST NOT send a PUBLISH packet containing a Topic Alias;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: If the Server accepts a connection with Clean Start set to 1;\nConsequent: the Server MUST set Session Present to 0 in the CONNACK packet in addition to setting a 0x00 (Success) Reason Code in the CONNACK packet;")
y.append("Antecedent: A PUBLISH packet sent from a Client to a Server;\nConsequent: MUST NOT contain a Subscription Identifier;")
y.append("Antecedent: The SUBACK packet sent by the Server to the Client;\nConsequent: MUST contain a Reason Code for each Topic Filter/Subscription Option pair;")
y.append("Antecedent: packets exceeding Maximum Packet Size;\nConsequent: The Server MUST NOT send packets to the Client;")
# 30
y.append("Antecedent: In the case of an error in any other packet, before closing the Network Connection;\nConsequent: it SHOULD send a DISCONNECT packet containing the Reason Code;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: If the value of Retain As Published subscription option is set to 0, when forwarding an Application Message;\nConsequent: the Server MUST set the RETAIN flag to 0;")
y.append("Antecedent: If a Server sends a CONNACK packet containing a Reason code of 128 or greater;\nConsequent: it MUST then close the Network Connection;")
y.append("Antecedent: If the Server receives a PUBLISH packet with the RETAIN flag set to 1, and QoS 0;\nConsequent: it SHOULD store the new QoS 0 message as the new retained message for that topic, but MAY choose to discard it at any time;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Topic Alias values greater than 0 and less than or equal to the Topic Alias Maximum value;\nConsequent: A Server MUST accept all Topic Alias that it returned in the CONNACK packet;")
y.append("Antecedent: Where a Packet is too large to send;\nConsequent: the Server MUST discard it without sending it and then behave as if it had completed sending that Application Message;")
# 40
y.append("Antecedent: If the Client's Session terminates before the Client reconnects;\nConsequent: the Server MUST NOT send the Application Message to any other subscribed Client;")
y.append("Antecedent: In this case;\nConsequent: the Server MUST deliver the message to the Client respecting the maximum QoS of all the matching subscriptions;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: when forwarding the Application Message;\nConsequent: The Server MUST maintain the order of User Properties;")
y.append("Antecedent: If the Authentication Method selected by the Client specifies that the Client sends data first;\nConsequent: the Client SHOULD include an Authentication Data property in the CONNECT packet;")
y.append("Antecedent: for which it has not received PUBACK, PUBCOMP, or PUBREC with a Reason Code of 128 or greater from the Server;\nConsequent: The Client MUST NOT send more than Receive Maximum QoS 1 and QoS 2 PUBLISH packets;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: The order of Reason Codes in the UNSUBACK packet;\nConsequent: MUST match the order of Topic Filters in the UNSUBSCRIBE packet;")
y.append("Antecedent: The Client or Server sending the DISCONNECT packet;\nConsequent: MUST use one of the DISCONNECT Reason Code values;")
y.append("Antecedent: In the case of an error in a AUTH packet, before closing the Network Connection;\nConsequent: it MAY send a DISCONNECT packet containing the reason code;")
# 50
y.append("Antecedent: subscribers receiving the Application Message;\nConsequent: The Server MUST send the Correlation Data unaltered to all subscribers;")
y.append("Antecedent: if it would increase the size of the DISCONNECT packet beyond the Maximum Packet Size specified by the receiver;\nConsequent: The sender MUST NOT send this property;")
y.append("Antecedent: while the Network Connection is open;\nConsequent: The Client and Server MUST NOT discard the Session State;")
y.append("Antecedent: If the Client does not receive a CONNACK packet from the Server within a reasonable amount of time;\nConsequent: the Client SHOULD close the Network Connection;")
y.append("Antecedent: If the RETAIN flag is set to 1 in a PUBLISH packet sent by a Client to a Server;\nConsequent: the Server MUST replace any existing retained message for this topic and store the Application Message [MQTT-3.3.1-5], so that it can be delivered to future subscribers whose subscriptions match its Topic Name;")
y.append("Antecedent: If the Server sends a single copy of the message;\nConsequent: it MUST include in the PUBLISH packet the Subscription Identifiers for all matching subscriptions which have a Subscription Identifiers, their order is not significant;")
y.append("Antecedent: The Topic Name in a PUBLISH packet sent by a Server to a subscribing Client;\nConsequent: MUST match the Subscription’s Topic Filter according to the matching process defined in section 4.7;")
y.append("Antecedent: If a CONNECT packet is received with Clean Start is set to 1;\nConsequent: the Client and Server MUST discard any existing Session and start a new Session;")
y.append("Antecedent: If the Keep Alive value is non-zero and the Server does not receive an MQTT Control Packet from the Client within one and a half times the Keep Alive time period;\nConsequent: it MUST close the Network Connection to the Client as if the network had failed;")
y.append("Antecedent: If a CONNECT packet is received with Clean Start set to 0 and there is a Session associated with the Client Identifier;\nConsequent: the Server MUST resume communications with the Client based on state from the existing Session;")
# 60
y.append("Antecedent: In the QoS 1 delivery protocol;\nConsequent: the sender MUST assign an unused Packet Identifier each time it has a new Application Message to publish;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: when it receives the first subscription or an Application Message with that Topic Name;\nConsequent: The topic resource MAY be either predefined in the Server by an administrator or it MAY be dynamically created by the Server;")

with open("../data/mqtt_sentence_condition_split", "wb") as file:
    pickle.dump(y, file)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model/condition_splitter.pt", map_location=device)

print(len(y))
data = []
for i in range(len(y)):
    data.append(f"<|startoftext|>Sentence: {sampled_mqtt_rule_sentences[i]}\n{y[i]}<|endoftext|>")

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

with open(r"../results/MQTT_condition_split_benchmark.txt", "a") as file:
    file.write(f"test loss: {average_test_loss} test bleu: {average_test_bleu}\n")
    file.write("\n")
