from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
from prepare_pretrain_data import prepare_pretrain_data
import re
from tqdm import tqdm
import pickle
from nltk.translate.bleu_score import sentence_bleu
import random

random.seed(4)


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
    train_decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_train = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    train_generated = []
    for i in range(len(decoded_train)):
        decoded_train[i] = [decoded_train[i].split("\n")[0].split("Context:")[1].strip(),
                            decoded_train[i].split("\n")[1].split("Sentence:")[1].strip()]
    for sample in decoded_train:
        train_generated.append(
            tokenizer.decode(generate(sample[0], sample[1], model, tokenizer)[0], skip_special_tokens=True))
    return train_loss, train_decoded_labels, train_generated


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
rule_condition_split = pickle.load(open('../data/coap_sentence_condition_split', 'rb'))

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
                contexts.append((context, split.split("Antecedent:")[1].strip(), "Antecedent rule", i))
            else:
                contexts.append((context, split.split("Consequent:")[1].strip(), "Consequent rule", i))

# for i in range(len(contexts)):
#     sentence = contexts[i][1]
#     if re.search(r"\b" + "unless" + r"\b", sentence) or re.search(r"\b" + "Unless" + r"\b", sentence):
#         print(i)

# The form of label is "Entity @ variable operator value;"
y = []
y.append("CoAP version number @ set to 1 = True;")
y.append("CoAP version number @ unknown = True;")
y.append("Messages @ be silently ignored = True;")
y.append("Token Length @ Lengths 9 to 15 = True;")
y.append("message @ be sent = True; Message @ be processed as a message format error = True;")
y.append("message @ presence of a marker followed by a zero length payload = True;")
y.append("message @ be processed as a message format error = True;")
y.append(
    "Option Numbers @ instances appear in order of their Option Numbers = True; Option Numbers @ delta encoding is used = True;")
y.append("Option Delta @ set to 15 = True; Option Delta @ entire byte is the payload marker = False;")
y.append("message @ be processed as a message format error = True;")
# 10
y.append("Option Length @ set to 15 = True;")
y.append("message @ be processed as a message format error = True;")
y.append("Option Value @ length and format define variable length values = True;")
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
y.append("CoAP endpoint @ keep track of timeout = True; CoAP endpoint @ keep track of retransmission counter = True;")
y.append("Retransmission @ the entire sequence of (re)transmissions stay in the envelope of MAX_TRANSMIT_SPAN = True;")
# 30
y.append("CoAP endpoint @ sent a Confirmable message = True;")
y.append("CoAP endpoint @ give up in attempting to obtain an ACK = True;")
y.append("responder @ needed = True;")
y.append(
    "responder @ rely on this cross-layer behavior from a requester = False; responder @ retain the state to create the ACK for the request = True;")
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
    "Non-confirmable message @ carries either a request or response = True; Non-confirmable message @ be Empty = False;")
y.append("Non-confirmable message @ be acknowledged by the recipient = False;")
y.append("recipient @ lacks context to process the message properly = True;")
# 50
y.append("recipient @ reject the message = True;")
y.append("recipient @ reject a Non-confirmable message = True;")
y.append("recipient @ send a matching Reset message = True; rejected message @ be ignored = True;")
y.append("sender @ transmit multiple copies of a Non-confirmable message within MAX_TRANSMIT_SPAN = True;")
y.append(
    "recipient @ echo Message ID in Acknowledgement message = True; recipient @ echo Message ID in Reset message = True;")
y.append("Message ID @ be reused within the EXCHANGE_LIFETIME = False;")
y.append(
    "Acknowledgement message @ match Confirmable message = True; Acknowledgement message @ match Non-confirmable message = True; Reset message @ match Confirmable message = True; Reset message @ match Non-confirmable message = True;")
y.append(
    "Acknowledgement message @ match Message ID of Confirmable message = True; Acknowledgement message @ match source endpoint with destination endpoint of Confirmable message = True; Acknowledgement message @ match Message ID of Non-confirmable message = True; Acknowledgement message @ match source endpoint with destination endpoint of Non-confirmable message = True; Reset message @ match Message ID of Confirmable message = True; Reset message @ match source endpoint with destination endpoint of Confirmable message = True; Reset message @ match Message ID of Non-confirmable message = True; Reset message @ match source endpoint with destination endpoint of Non-confirmable message = True;")
y.append(
    "recipient @ acknowledge each duplicate copy of a Confirmable message using the same Acknowledgement or Reset message = True; recipient @ process any request or response in the message only once = True;")
y.append(
    "Confirmable message @ transports a request that is idempotent = True; Confirmable message @ transports a request can be handled in an idempotent fashion = True;")
# 60
y.append(
    "recipient @ acknowledge each duplicate copy of a Confirmable message using the same Acknowledgement or Reset message = False; recipient @ process any request or response in the message only once = False;")
y.append(
    "recipient @ ignore any duplicated Non-confirmable messages = True; recipient @ process any request or response in the message only once = True;")
y.append("CoAP message @ appropriately encapsulated = True;")
y.append("CoAP message @ fit within a single IP packet = True; CoAP message @ fit within a single IP datagram = True;")
y.append("Path MTU @ known for a destination = False;")
y.append("IP MTU @ 1280 bytes be assumed = True;")
y.append("headers @ nothing is known about the size = True;")
y.append("message @ 1152 bytes size upper bound = True; payload @ 1024 bytes size upper bound = True;")
y.append("clients @ limit the number of simultaneous outstanding interactions to NSTART = True;")
y.append("algorithm @ modified by additional congestion control optimizations = False;")
# 70
y.append(
    "algorithm @ be chosen in such a way that an endpoint does not exceed an average data rate of PROBING_RATE = True;")
y.append("server @ implement some rate limiting for response transmissions = True;")
y.append("application environment @ use consistent values for these parameters = True;")
y.append("Configurations @ use mechanisms that ensure congestion control safety = False;")
y.append("Configurations @ decrease ACK_TIMEOUT = False; Configurations @ increase NSTART = False;")
y.append(
    "ACK_RANDOM_FACTOR @ be decreased below 1.0 = False; ACK_RANDOM_FACTOR @ have a value that is sufficiently different from 1.0 = True;")
y.append("Transmission parameters @ leads to an increase of derived time values = True;")
y.append("configuration mechanism @ ensure the adjusted value is also available to all the endpoints = True;")
y.append("GET method @ take any other action on a resource other than retrieval = False;")
y.append("GET method @ idempotent = True; PUT method @ idempotent = True; DELETE method @ idempotent = True;")
# 80
y.append(
    "client @ prepare to receive piggybacked responses = True; client @ prepare to receive separate responses = True;")
y.append("server @ send back an Empty Acknowledgement = True;")
y.append("server @ send back the response in another Acknowledgement = False;")
y.append("server @ receive retransmitted requests = True;")
y.append("server @ send another Empty Acknowledgement = True; server @ send response as a separate response = True;")
y.append("server @ send a Confirmable response = True;")
y.append("client @ Acknowledgement be an Empty message = True;")
y.append("server @ matching Acknowledgement message = True; server @ matching Reset message = True;")
y.append("server @ retransmit response = False;")
y.append("request message @ Non-confirmable = True;")
# 90
y.append("response @ Non-confirmable message = True;")
y.append(
    "endpoint @ receive a Non-confirmable response in reply to a Confirmable request = True; endpoint @ receive a Confirmable response in reply to a Non-confirmable request = True;")
y.append("server @ echo client-generated token in response = True;")
y.append("client @ generate unique tokens = True;")
y.append("client @ send a request without using Transport Layer Security = True;")
y.append("client @ use a nontrivial randomized token = True;")
y.append("client @ connected to the general Internet = True;")
y.append("client @ use at least 32 bits of randomness = True;")
y.append("endpoint @ receive a token it did not generate = True;")
y.append(
    "endpoint @ treat the token as opaque = True; endpoint @ make no assumptions about its content or structure = True;")
# 100
y.append("response @ source endpoint be the same as the destination endpoint of the original request = True;")
y.append(
    "piggybacked response @ Message ID of the Confirmable request and the Acknowledgement match = True; piggybacked response @ tokens of the response and original request match = True;")
y.append("separate response @ tokens of the response and original request match = True;")
y.append("option @ not defined for a Method or Response Code = True;")
y.append("sender @ include option = False; recipient @ treat option like an unrecognized option = True;")
y.append('endpoint @ receive unrecognized options of class "elective" = True;')
y.append('endpoint @ ignore unrecognized options of class "elective" = True;')
y.append('"critical" unrecognized options @ occur in Confirmable request = True;')
y.append('"critical" unrecognized options @ return 4.02 (Bad Option) response = True;')
y.append("4.02 (Bad Option) response @ include a diagnostic payload = True;")
# 110
y.append(
    '"critical" unrecognized options @ occur in Confirmable response = True; "critical" unrecognized options @ piggybacked in an Acknowledgement = True;')
y.append("response @ rejected = True;")
y.append('"critical" unrecognized options @ occur in Non-confirmable message = True;')
y.append("message @ rejected = True;")
y.append("option length @ outside the defined range in a request = True;")
y.append("option @ be treated like an unrecognized option = True;")
y.append("option value @ default value = True;")
y.append("option @ be included in message = False;")
y.append("option @ present = False;")
y.append("option value @ default value = True;")
# 120
y.append("option @ repeatable = True;")
y.append("option @ be included one or more times in a message = True;")
y.append("option @ repeatable = False;")
y.append("option @ be included more than once in a message = False;")
y.append("message @ includes an option with more occurrences than the option is defined for = True;")
y.append("option @ supernumerary option occurrence be treated like an unrecognized option = True;")
y.append("Method Code @ defined to have a payload = False; Response Code @ defined to have a payload = False;")
y.append("sender @ include a payload = False; recipient @ ignore the payload = True;")
y.append("Content type @ given = False;")
y.append('Payload "sniffing" @ be attempted = True;')
# 130
y.append("requirement @ protocol requirement = False;")
y.append("diagnostic message @ encoded using UTF-8 = True;")
y.append("payload @ additional information beyond the Response Code = False;")
y.append("payload @ be empty = True;")
y.append("Caching CoAP endpoint @ cache responses = True;")
y.append("Response Codes @ indicate success = True; Response Codes @ unrecognized by an endpoint = True;")
y.append("Response Codes @ be cached = False;")
y.append(
    "Request method @ match that used to obtain stored response = False; options @ match options in request used to obtain stored response = False; stored response @ fresh or validated = False;")
y.append("CoAP endpoint @ use a stored response = False;")
y.append("original server @ wishes to prevent caching = True;")
# 140
y.append("original server @ include a Max-Age Option with a value of zero seconds = True;")
y.append("endpoint @ use ETag in GET request = True;")
y.append("endpoint @ add an ETag Option = True;")
y.append("response @ be used to satisfy the request = True; response @ replace the stored response = True;")
y.append("client @ use a proxy to make a request that will use a secure URI scheme = True;")
y.append("request @ be sent using DTLS = True;")
y.append("request @ time out = True;")
y.append("5.04 (Gateway Timeout) response @ be returned = True;")
y.append("request @ response returned cannot be processed by the proxy = True;")
y.append("5.02 (Bad Gateway) response @ be returned = True;")
# 150
y.append("response @ generated out of a cache = True;")
y.append("Max-Age Option @ extend the max-age originally set by the server = False;")
y.append("options @ present in proxy request = True;")
y.append("options @ be processed at the proxy = True;")
y.append("Unsafe options @ in a request not recognized by the proxy = True;")
y.append("4.02 (Bad Option) response @ be returned = True;")
y.append("CoAP-to-CoAP proxy @ recognize Safe-to-Forward options = False;")
y.append("CoAP-to-CoAP proxy @ forward to the origin server all Safe-to-Forward options = True;")
y.append("CoAP-to-CoAP proxy @ recognize Unsafe options = False;")
y.append("5.02 (Bad Gateway) response @ be returned = True;")
# 160
y.append("Safe-to-Forward options @ not recognized = True;")
y.append("Safe-to-Forward options @ be forwarded = True;")
y.append(
    "endpoint @ receive a proxy request = True; endpoint @ unwilling or unable to act as proxy for the request URI = True;")
y.append("5.05 (Proxying Not Supported) response @ be returned = True;")
y.append("authority @ recognized as proxy endpoint itself = True;")
y.append("request @ be treated as a local request = True;")
y.append("proxy @ configured to forward the proxy request to another proxy = False;")
y.append(
    "proxy @ translate the request as follows: the scheme of the request URI defines the outgoing protocol and its details = True;")
y.append("request @ with an unrecognized Method Code = True; request @ with an unsupported Method Code = True;")
y.append("4.05 (Method Not Allowed) response @ be returned = True;")
# 170
y.append("GET @ success= True;")
y.append("2.05 (Content) response @ be returned = True; 2.03 (Valid) response @ be returned = True;")
y.append("resource @ be created on the server = True;")
y.append(
    "2.01 (Created) response @ be returned = True; response @ include the URI of the new resource in a sequence of one or more Location-Path and/or Location-Query Options = True;")
y.append("POST @ succeed= True; resource @ be created on the server = False;")
y.append("2.04 (Changed) response @ be returned = True;")
y.append("POST @ succeed = True; target resource @ be deleted = True;")
y.append("2.02 (Deleted) response @ be returned = True;")
y.append("resource @ exists at the request URI = True;")
y.append(
    "representation @ be considered a modified version of that resource = True; 2.04 (Changed) response @ be returned = True;")
# 180
y.append("resource @ exists = False;")
y.append("server @ create a new resource with that URI = True; 2.01 (Created) response @ be returned = True;")
y.append("resource @ be created = False; resource @ be modified = False;")
y.append("Response Code @ send error = True;")
y.append("DELETE @ succeed = True; resource @ exist before request = False;")
y.append("2.02 (Deleted) response @ be returned = True;")
y.append("cache @ receive response includes one or more Location-Path and/or Location-Query Options = True;")
y.append("cache @ mark any stored response for the created resource as not fresh = True;")
y.append("cache @ mark any stored response for the deleted resource as stale = True;")
y.append("response @ include an ETag Option = True; response @ include a payload = False;")
# 190
y.append(
    "cache @ recognizes and processes the ETag response option = True; 2.03 (Valid) response @ be returned = True;")
y.append("cache @ update the stored response with the value of the Max-Age Option included in the response = True;")
y.append(
    "Safe-to-Forward option @ present in the response = True; Safe-to-Forward option @ present in the stored response = True;")
y.append("Safe-to-Forward option @ be replaced with the set of options of this type in the response received = True;")
y.append("cache @ mark stored response for the changed resource as not fresh = True;")
y.append("server @ include a diagnostic payload = True;")
y.append("client @ improve authentication status to the server = False;")
y.append("client @ repeat the request = False;")
y.append("client @ modification = False;")
y.append("client @ repeat the request = False;")
# 200
y.append("server @ in a position to make this information available = True;")
y.append("response @ include a Size1 Option = True;")
y.append("server @ include a diagnostic payload = True;")
y.append("Uri-Path Option @ value be '.' = False;")
y.append(
    "forward-proxy @ forward the request on to another proxy = True; forward-proxy @ forward the request directly to the server specified by the absolute-URI = True;")
y.append("proxy @ be able to recognize all of its server names = True;")
y.append(
    "endpoint @ receiving a request with a Proxy-Uri Option that is unable or unwilling to act as a forward-proxy for the request = True;")
y.append("5.05 (Proxying Not Supported) response @ be returned = True;")
y.append(
    "Proxy-Uri Option @ take precedence over Uri-Host = True; Proxy-Uri Option @ take precedence over Uri-Port = True; Proxy-Uri Option @ take precedence over Uri-Path = True; Proxy-Uri Option @ take precedence over Uri-Query = True;")
y.append(
    "Content-Format @ preferred Content-Format returned = False; response @ another error code takes precedence = False;")
# 210
y.append('4.06 "Not Acceptable" response @ be returned = True;')
y.append("Servers @ provide resources with strict tolerances on the value of Max-Age = True;")
y.append("Servers @ update the value before each retransmission = True;")
y.append("endpoint @ receiving an entity-tag = True;")
y.append("endpoint @ treat entity-tag as opaque and make no assumptions about its content or structure = True;")
y.append("ETag Option @ occur more than once in a response = False;")
y.append("ETag Option @ occur zero, one, or multiple times in a request = True;")
y.append(
    "response @ with one or more Location-Path and/or Location-Query Options passes through a cache = True; cache @ interprets these options and the implied URI identifies one or more currently stored responses = True;")
y.append("stored responses @ be marked as not fresh = True;")
y.append("Location-Path Option @ value be '.' = False;")
# 220
y.append(
    "reserved option numbers @ occur in addition to Location-Path and/or Location-Query = True; reserved option numbers @ supported by the server = False;")
y.append("4.02 (Bad Option) response @ be returned = True;")
y.append("Conditional request options @ condition given is fulfilled = False;")
y.append("server @ perform the requested method = False;")
y.append("Conditional request options @ condition given is fulfilled = False;")
y.append("4.12 (Precondition Failed) @ be returned = True;")
y.append(
    "request @ with conditional request options = False; 2.xx response @ be returned = False; 4.12 Response Code @ be returned = False;")
y.append("conditional request options @ be ignored = True;")
y.append("ETag @ exist = True; ETag @ value for one or more representations of the target resource = True;")
y.append("If-Match Option @ be used to make a request = True;")
# 230
y.append("target resource @ exist = False;")
y.append("If-None-Match Option @ be used to make a request = True;")
y.append("URI @ has authority = False; URI @ has host = False;")
y.append("URI @ invalid = True;")
y.append("port subcomponent @ empty = True; port subcomponent @ given = False;")
y.append("UDP port 5684 @ be assumed = True; UDP datagrams @ be secured through the use of DTLS = True;")
y.append("percent-encoding @ use uppercase letters = True;")
y.append("server @ offers resources for resource discovery = True;")
y.append(
    "server @ support CoAP port number 5683 = True; port 5683 @ be supported for providing access to other resources = True;")
y.append(
    "server @ support port number 5684 = True; port 5684 @ be supported for resource discovery = True; port 5684 @ be supported for providing access to other resources = True;")
# 240
y.append("configuration @ fully manual configuration is desired = False;")
y.append("CoAP endpoint @ support the CoRE Link Format of discoverable resources = True;")
y.append("CoAP identifier code format @ be in the range of 0-65535 = True;")
y.append("Content-Format code @ include a space-separated sequence of Content-Format codes = True;")
y.append("multicast service discovery @ is desired = False;")
y.append(
    "endpoint @ be prepared to receive multicast requests on other multicast addresses = True; endpoint @ process multicast requests on other multicast addresses = False;")
y.append("multicast requests @ be addressed to an IP multicast address = True;")
y.append("multicast requests @ be Non-confirmable = True;")
y.append("server @ available = True;")
y.append("server @ be aware that a request arrived via multicast = True;")
# 250
y.append("server @ be aware that a request arrived via multicast = True;")
y.append("server @ return a Reset message in reply to a Non-confirmable message = False;")
y.append("server @ be aware that a request arrived via multicast = False;")
y.append("server @ return a Reset message in reply to a Non-confirmable message = True;")
y.append("sender @ using a Message ID that is also still active = False;")
y.append(
    "server @ is aware that a request arrived via multicast = True; server @ have anything useful to respond = False;")
y.append("server @ ignore the request = True;")
y.append("Leisure @ value be derived = True;")
y.append(
    "server @ pick a random point of time within the chosen leisure period to send back the unicast response to the multicast request = True;")
y.append("CoAP endpoint @ have suitable data to compute a value for Leisure = False;")
# 260
y.append("Leisure @ default value = True;")
y.append("response @ match to multicast request = True;")
y.append("token @ match = True; source endpoint @ match destination endpoint = False;")
y.append("cache @ update with the received responses = True;")
y.append("response @ received in reply to a GET request to a multicast group = True;")
y.append("response @ be used to satisfy a subsequent request on the related unicast request URI = True;")
y.append("cache @ revalidate a response by making a GET request on the related unicast request URI = True;")
y.append("GET @ to multicast group = True;")
y.append("GET @ contain ETag option = True;")
y.append("lower-layer security @ be used when appropriate = True;")
# 270
y.append('CoAP message @ be sent as DTLS "application data" = True;')
y.append(
    "rules @ added for matching an Acknowledgement message or Reset message to a Confirmable message = True; rules @ added for a Reset message to a Non-confirmable message = True;")
y.append("DTLS session @ be the same = True; epoch @ be the same = True;")
y.append("Retransmissions @ be performed across epochs = False;")
y.append(
    "rules @ added for matching a response to a request = True; DTLS session @ be the same = True; epoch @ be the same = True;")
y.append("response @ to a DTLS secured request = True;")
y.append("response @ be DTLS secured using the same security session and epoch = True;")
y.append("NoSec response @ to a DTLS request = True;")
y.append("NoSec response @ be rejected = True;")
y.append("Endpoint Identity Devices @ support the Server Name Indication (SNI) = True;")
# 280
y.append("Implementations @ support the mandatory-to-implement cipher suite TLS_PSK_WITH_AES_128_CCM_8 = True;")
y.append("device @ be configured with multiple raw public keys = True;")
y.append("RawPublicKey @ support the mandatory-to-implement cipher suite TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8 = True;")
y.append("key @ be ECDSA capable = True;")
y.append("curve secp256r1 @ be supported = True;")
y.append(
    "Implementations @ use the Supported Elliptic Curves = True; Implementations @ use the Supported Point Formats = True;")
y.append("implementations @ support checking RawPublicKey identities = True;")
y.append("implementations @ support at least the sha-256-120 mode (SHA-256 truncated to 120 bits) = True;")
y.append(
    "Implementations @ support longer length identifiers = True; implementations @ support shorter lengths = True;")
y.append("implementations @ support shorter lengths = False;")
# 290
y.append("implementations @ have a user interface = True;")
y.append(
    "implementations @ support the binary mode = True; implementations @ support the human-speakable format = True;")
y.append("access control list of identifiers @ during provisioning = True;")
y.append(
    "access control list of identifiers @ be installed = True; access control list of identifiers @ be maintained = True;")
y.append("Implementations @ in Certificate mode = True;")
y.append("Implementations @ support the mandatory-to-implement cipher suite TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8 = True;")
y.append("Certificates @ be signed with ECDSA using secp256r1 = True; signature @ use SHA-256 = True;")
y.append("key @ be ECDSA capable = True;")
y.append("curve secp256r1 @ be supported = True;")
y.append(
    "Implementations @ use the Supported Elliptic Curves = True; Implementations @ use the Supported Point Formats = True;")
# 300
y.append("CoAP node @ has a source of absolute time = True;")
y.append("CoAP node @ check that the validity dates of the certificate are within range = True;")
y.append("certificate @ be validated as appropriate for the security requirements = True;")
y.append("certificate @ contains a SubjectAltName = True;")
y.append(
    "request URI @ authority match at least one of the authorities of any CoAP URI found in a field of URI type in the SubjectAltName set = True;")
y.append("certificate @ contains a SubjectAltName = False;")
y.append("request URI @ authority match the Common Name (CN) found in the certificate = True;")
y.append("system @ has a shared key in addition to the certificate = True;")
y.append("cipher suite @ includes the shared key be used = True;")
y.append(
    "underlying model of Confirmable messages @ have no effect on a proxy function = True; underlying model of Non-confirmable messages @ have no effect on a proxy function = True;")
# 310
y.append("GET @ success = True;")
y.append("2.05 (Content) @ be returned = True;")
y.append("payload @ response = True;")
y.append(
    "payload @ be a representation of the target HTTP resource = True; Content-Format Option @ be set accordingly = True;")
y.append(
    "response @ indicate a Max-Age value that is no greater than the remaining time the representation can be considered fresh = True;")
y.append("HTTP entity @ has an entity-tag = True;")
y.append("proxy @ include an ETag Option in the response = True; proxy @ process ETag Options in requests = True;")
y.append("request @ include an Accept Option = True;")
y.append("request @ include one or more ETag Options = True;")
y.append("new resource @ is created at the request URI = True;")
# 320
y.append("2.01 (Created) response @ be returned = True;")
y.append("existing resource @ is modified = True;")
y.append("2.04 (Changed) response @ be returned = True;")
y.append("DELETE @ success = True; resource @ exist = False;")
y.append("2.02 (Deleted) response @ be returned = True;")
y.append("POST @ result in a resource that can be identified by a URI = False;")
y.append("2.04 (Changed) response @ be returned = True;")
y.append("resource @ be created on the origin server = True;")
y.append("2.01 (Created) response @ be returned = True;")
y.append("statements @ RECOMMENDED behaviour = True;")
# 330
y.append("OPTIONS method @ supported in CoAP = False; TRACE method @ supported in HTTP = False;")
y.append("501 (Not Implemented) error @ be returned = True;")
y.append(
    "payload of response @ be a representation of the target CoAP resource = True; Content-Type @ be set accordingly = True; Content-Encoding @ be set accordingly = True;")
y.append(
    "response @ indicate a max-age directive that indicates a value no greater than the remaining time the representation can be considered fresh = True;")
y.append("proxy @ retry the request with further media types from the HTTP Accept header field = True;")
y.append("HEAD method @ identical to GET = True; server @ return a message-body in the response = False;")
y.append("POST method @ result in a resource that can be identified by a URI = False;")
y.append("200 (OK) response @ be returned = True; 204 (No Content) response @ be returned = True;")
y.append("resource @ be created on the origin server = True;")
y.append("201 (Created) response @ be returned = True;")
# 340
y.append(
    "caching proxy @ make cached values available to requests that have lesser transport-security properties than those the proxy would require to perform request forwarding in the first place = False;")
y.append(
    "cache @ is able to make equivalent access control decisions to the ones that led to the cached entry = False;")
y.append('responses to "coaps" @ be reused for shared caching = False;')
y.append("request @ is authenticated = False;")
y.append("large amplification factors @ be provided in the response = False;")
y.append("multicast requests @ be authenticated = False;")
y.append("CoAP servers @ accept multicast requests = False;")
y.append("CoAP server @ possible = True;")
y.append("CoAP server @ limit the support for multicast requests = True;")
y.append("Implementation @ available = True;")
# 350
y.append("Implementation @ make use of modern APIs = True;")
y.append("constrained nodes @ good source of entropy = False;")
y.append("node @ be used for processes that require good entropy = False;")
y.append("option numbers @ between 65000 and 65535 = True; option numbers @ be used in operational deployments = True;")
y.append("option numbers @ between 65000 and 65535 = True; option numbers @ be used in operational deployments = True;")

with open("../data/coap_split_properties", "wb") as file:
    pickle.dump(y, file)

with open("../data/coap_contexts_condition_split", "wb") as file:
    pickle.dump(contexts, file)
#
#
#
print(len(y))
# data = []
# for i in range(len(y)):
#     data.append(
#         f"<|startoftext|>Context: {contexts[i][0]}\nSentence: {contexts[i][1]}\nProperties: {y[i]}<|endoftext|>")
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
#                                           pad_token="<|pad|>")
# model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# model.resize_token_embeddings(len(tokenizer))
#
# inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
# labels = inputs["input_ids"].clone()
# inputs["labels"] = labels
#
# dataset = MeditationDataset(inputs)
# dataset_length = len(dataset)
# train_length = int(dataset_length * 0.9)
# test_length = dataset_length - train_length
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
#
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# optimizer = AdamW(model.parameters(), lr=2e-5)
#
# for epoch in range(100):
#     train_loop = tqdm(train_loader, leave=True)
#     overall_train_loss = 0
#     num_of_train_batches = len(train_loader)
#     overall_train_bleu = 0
#     for train_batch in train_loop:
#         train_batch_bleu = 0
#         model.train()
#         train_loss, train_decoded_labels, train_generated = train(train_batch, model, optimizer)
#         train_loop.set_postfix(train_loss=train_loss.item())
#         overall_train_loss += train_loss.item()
#         train_loop.set_description(f"Epoch {epoch} train")
#
#         for i in range(len(train_decoded_labels)):
#             train_decoded_labels[i] = train_decoded_labels[i].split("Properties:", 1)[1].strip()
#             train_generated[i] = train_generated[i].split("Properties:", 1)[1].strip()
#
#         for i in range(len(train_decoded_labels)):
#             train_batch_bleu += sentence_bleu([train_decoded_labels[i]], train_generated[i])
#
#         overall_train_bleu += train_batch_bleu / len(train_decoded_labels)
#
#     test_loop = tqdm(test_loader, leave=True)
#     overall_test_loss = 0
#     num_of_test_batches = len(test_loader)
#     overall_test_bleu = 0
#     for test_batch in test_loop:
#         test_batch_bleu = 0
#         model.eval()
#         test_loss, test_decoded_labels, test_generated = test(test_batch, model)
#         test_loop.set_postfix(test_loss=test_loss.item())
#         overall_test_loss += test_loss.item()
#         test_loop.set_description(f"Epoch {epoch} test")
#
#         for i in range(len(test_decoded_labels)):
#             test_decoded_labels[i] = test_decoded_labels[i].split("Properties:", 1)[1].strip()
#             test_generated[i] = test_generated[i].split("Properties:", 1)[1].strip()
#
#         for i in range(len(test_decoded_labels)):
#             test_batch_bleu += sentence_bleu([test_decoded_labels[i]], test_generated[i])
#         overall_test_bleu += test_batch_bleu
#
#     average_train_loss = overall_train_loss / num_of_train_batches
#     print(f"average train loss: {average_train_loss}")
#     average_test_loss = overall_test_loss / num_of_test_batches
#     print(f"average test loss: {average_test_loss}")
#     average_train_bleu = overall_train_bleu / num_of_train_batches
#     print(f"average train bleu: {average_train_bleu}")
#     average_test_bleu = overall_test_bleu / num_of_test_batches
#     print(f"average test bleu: {average_test_bleu}")
#
#     with open(r"../results/properties_extractor_results.txt", "a") as file:
#         file.write(
#             f"Epoch {epoch} average_train_loss: {average_train_loss} average train bleu: {average_train_bleu}")
#         file.write("\n")
#         file.write(
#             f"Epoch {epoch} average_test_loss: {average_test_loss} average test bleu: {average_test_bleu}")
#         file.write("\n")
#
#     if average_train_loss < 0.1 and average_test_loss < 0.3:
#         break
#
# torch.save(model, "../model/properties_extractor.pt")
