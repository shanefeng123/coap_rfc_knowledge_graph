from transformers import PegasusForConditionalGeneration, PegasusTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from prepare_pretrain_data import prepare_pretrain_data
import re
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
    return train_loss


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    test_loss = test_outputs.loss
    return test_loss


def generate(sentence, model, tokenizer):
    data = f"<|startoftext|>Sentence: {sentence}\nAntecedent:"
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=100)
    return generated


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
y = []
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Messages with unknown version numbers;\nConsequent: MUST be silently ignored;")
y.append(
    "Antecedent: Lengths 9-15 are reserved;\nConsequent: MUST NOT be sent, and MUST be processed as a message format error;")
y.append(
    "Antecedent: The presence of a marker followed by a zero-length payload;\nConsequent: MUST be processed as a message format error;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the field is set to this value but the entire byte is not the payload marker;\nConsequent: this MUST be processed as a message format error;")
y.append("Antecedent: If the field is set to this value;\nConsequent: it MUST be processed as a message format error;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: See Section 3.2 for the formats used in this document; options defined in other documents;\nConsequent: MAY make use of other option value formats;")
y.append(
    "Antecedent: if it has a choice;\nConsequent: An option definition may specify a range of permissible numbers of bytes; a sender SHOULD represent the integer with as few bytes as possible, i.e., without leading zero bytes;")
# 10
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: If there are any bytes;\nConsequent: they MUST be processed as a message format error;")
y.append(
    "Antecedent: if the recipient lacks context to process the message properly, including situations where the message is Empty, uses a code with a reserved class (1, 6, or 7), or has a message format error;\nConsequent: A recipient MUST either (a) acknowledge a Confirmable message with an Acknowledgement message or (b) reject the message;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: More generally, recipients of Acknowledgement and Reset messages;\nConsequent: MUST NOT respond with either Acknowledgement or Reset messages;")
y.append(
    "Antecedent: while waiting for an acknowledgement (or reset);\nConsequent: Retransmission is controlled by two things that a CoAP endpoint MUST keep track of for each Confirmable message it sends: a timeout and a retransmission counter;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: A CoAP endpoint that sent a Confirmable message;\nConsequent: MAY give up in attempting to obtain an ACK even before the MAX_RETRANSMIT counter value is reached;")
# 20
y.append(
    "Antecedent: if needed;\nConsequent: However, a responder MUST NOT in turn rely on this cross-layer behavior from a requester, i.e., it MUST retain the state to create the ACK for the request, even if a Confirmable response was already acknowledged by the requester;")
y.append("Antecedent: MAY be the receipt of ICMP errors;\nConsequent: Another reason for giving up retransmission;")
y.append(
    "Antecedent: If it is desired to take account of ICMP errors;\nConsequent: to mitigate potential spoofing attacks, implementations SHOULD take care to check the information about the original datagram in the ICMP message, including port numbers and CoAP header information such as message type and code, Message ID, and Token;\nAntecedent: if this is not possible due to limitations of the UDP service API;\nConsequent: ICMP errors SHOULD be ignored;")
y.append(
    'Antecedent: if the implementation note in Section 4.6 is followed;\nConsequent: Packet Too Big errors [RFC4443] ("fragmentation needed and DF set" for IPv4 [RFC0792]) cannot properly occur and SHOULD be ignored;\nAntecedent: otherwise;\nConsequent: they SHOULD feed into a path MTU discovery algorithm [RFC4821];')
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: after appropriate vetting;\nConsequent: Host, network, port, or protocol unreachable errors or parameter problem errors MAY be be used to inform the application of a failure in sending;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: if it lacks context to process the message properly, including the case where the message is Empty, uses a code with a reserved class (1, 6, or 7), or has a message format error;\nConsequent: A recipient MUST reject the message;")
y.append(
    "Antecedent: Rejecting a Non-confirmable message;\nConsequent: MAY involve sending a matching Reset message, and apart from the Reset message the rejected message MUST be silently ignored;")
# 30
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: For an Acknowledgement or Reset message to match a Confirmable or Non-confirmable message;\nConsequent: the Message ID and source endpoint of the Acknowledgement or Reset message MUST match the Message ID and destination endpoint of the Confirmable or Non-confirmable message;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: in case the Confirmable message transports a request that is idempotent (see Section 5.1) or can be handled in an idempotent fashion;\nConsequent: This rule MAY be relaxed;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: appropriately encapsulated;\nConsequent: A CoAP message SHOULD fit within a single IP packet (i.e., avoid IP fragmentation) and (by fitting into one UDP payload) obviously needs to fit within a single IP datagram；")
y.append(
    "Antecedent: If the Path MTU is not known for a destination;\nConsequent: an IP MTU of 1280 bytes SHOULD be assumed;\nAntecedent: if nothing is known about the size of the headers;\nConsequent: good upper bounds are 1152 bytes for the message size and 1024 bytes for the payload size;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 40
y.append(
    "Antecedent: Unless this is modified by additional congestion control optimizations;\nConsequent: it MUST be chosen in such a way that an endpoint does not exceed an average data rate of PROBING_RATE in sending to another endpoint that does not respond;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: without using mechanisms that ensure congestion control safety, either defined in the configuration or in future standards documents;\nConsequent: Configurations MUST NOT decrease ACK_TIMEOUT or increase NSTART;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the choice of transmission parameters leads to an increase of derived time values (see Section 4.8.2);\nConsequent: the configuration mechanism MUST ensure the adjusted value is also available to all the endpoints with which these adjusted values are to be used to communicate;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: Once the server sends back an Empty Acknowledgement;\nConsequent: it MUST NOT send back the response in another Acknowledgement, even if the client retransmits another identical request;")
# 50
y.append(
    "Antecedent: If a retransmitted request is received (perhaps because the original Acknowledgement was delayed);\nConsequent: another Empty Acknowledgement is sent, and any response MUST be sent as a separate response;")
y.append(
    "Antecedent: If the server then sends a Confirmable response;\nConsequent: the client\'s Acknowledgement to that response MUST also be an Empty message (one that carries neither a request nor a response);")
y.append(
    "Antecedent: on any matching Acknowledgement (silently ignoring any Response Code or payload) or Reset message;\nConsequent: The server MUST stop retransmitting its response;")
y.append(
    "Antecedent: If the request message is Non-confirmable;\nConsequent: then the response SHOULD be returned in a Non-confirmable message as well;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: A client sending a request without using Transport Layer Security (Section 9);\nConsequent: SHOULD use a nontrivial, randomized token to guard against spoofing of responses (Section 11.4);")
y.append(
    "Antecedent: A client that is connected to the general Internet;\nConsequent: SHOULD use at least 32 bits of randomness, keeping in mind that not being directly connected to the Internet is not necessarily sufficient protection against spoofing;")
y.append(
    "Antecedent: An endpoint receiving a token it did not generate;\nConsequent: MUST treat the token as opaque and make no assumptions about its content or structure;")
# 60
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: In case an option is not defined for a Method or Response Code;\nConsequent: it MUST NOT be included by a sender and MUST be treated like an unrecognized option by a recipient;")
y.append(
    'Antecedent: The difference between these is how an option unrecognized by an endpoint is handled: o Upon reception;\nConsequent: unrecognized options of class "elective" MUST be silently ignored;')
y.append(
    'Antecedent: Unrecognized options of class "critical" that occur in a Confirmable request;\nConsequent: MUST cause the return of a 4.02 (Bad Option) response;')
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    'Antecedent: Unrecognized options of class "critical" that occur in a Confirmable response, or piggybacked in an Acknowledgement;\nConsequent: MUST cause the response to be rejected (Section 4.2);')
y.append(
    'Antecedent: Unrecognized options of class "critical" that occur in a Non-confirmable message;\nConsequent: MUST cause the message to be rejected (Section 4.3);')
y.append(
    "Antecedent: If the length of an option value in a request is outside the defined range;\nConsequent: that option MUST be treated like an unrecognized option (see Section 5.4.1);")
# 70
y.append("Antecedent: If the value of an option is intended to be this default value;\nConsequent: the option SHOULD NOT be included in the message;")
y.append("Antecedent: If the option is not present;\nConsequent: the default value MUST be assumed;")
y.append("Antecedent: An option that is repeatable;\nConsequent: MAY be included one or more times in a message;")
y.append("Antecedent: An option that is not repeatable;\nConsequent: MUST NOT be included more than once in a message;")
y.append("If a message includes an option with more occurrences than the option is defined for;\nConsequent: each supernumerary option occurrence that appears subsequently in the message MUST be treated like an unrecognized option (see Section 5.4.1);")
y.append("If a Method or Response Code is not defined to have a payload;\nConsequent: then a sender MUST NOT include one, and a recipient MUST ignore it;")
y.append('Antecedent: if no content type is given;\nConsequent: Payload "sniffing" SHOULD only be attempted;')


print(len(y))
data = []
for i in range(len(y)):
    data.append(f"<|startoftext|>Sentence: {rfc7252_rule_sentences[i]}\n{y[i]}<|endoftext|>")

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

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

print(tokenizer.decode(generate(rfc7252_rule_sentences[74], model, tokenizer)[0]))

# for sentence in rfc7252_rule_sentences:
#     if re.search(r"\b" + "shorter" + r"\b", sentence):
#         print(sentence)

# conditional_sentences = []
# for sentence in rfc7252_rule_sentences:
#     for keyword in CONDITIONAL_KEYWORDS:
#         if keyword == "if":
#             if re.search(r"\b" + keyword.capitalize() + r"\b", sentence) or (
#                     re.search(r"\b" + keyword + r"\b", sentence) and
#                     not re.search(r"\b" + "even if" + r"\b", sentence)):
#                 print(sentence)
#
# model_name = "tuner007/pegasus_paraphrase"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name)
# model.to(device)
#
# batch = tokenizer(["""Messages with unknown version numbers MUST be silently ignored.
# """], truncation=True, padding="longest",
#                   return_tensors="pt").to(device)
#
# outputs = tokenizer.batch_decode(
#     model.generate(**batch, max_length=100, num_beams=10, num_return_sequences=5, temperature=1.5))
