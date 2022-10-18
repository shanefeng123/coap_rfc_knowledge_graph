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
# Annotate 100 samples
y = []
y.append("set this field to 1 = True;")
y.append("Message with unknown version numbers = True; message be silently ignored = True;")
y.append("Lengths 9-15 = True; message be sent = False; message be processed as a message format error = True;")
y.append(
    "presence of a marker followed by a zero-length payload = True; message be processed as a message format error = True;")
y.append("instances appear in order of their Option Numbers = True; delta encoding is used = True;")
y.append(
    "the field is set to this value = True; entire byte is the payload marker = False; message be processed as a message format error = True;")
y.append("the field is set to this value = True; message be processed as a message format error = True;")
y.append("length and format define variable-length values = True;")
y.append("options defined in other documents make use of other option value formats = True;")
y.append("it has a choice = True; sender represent the integer with as few bytes as possible = True;")
y.append("recipient be prepared to process values with leading zero bytes = True;")
y.append("Token Length field be set to 0 = True; bytes of data be present after the Message ID field = False;")
y.append("any bytes = True; message be processed as a message format error = True;")
y.append(
    "recipient acknowledge a Confirmable message with an Acknowledgement message = True; recipient reject the message = True; recipient lacks context to process the message properly = True;")
y.append(
    "Acknowledgement message echo the Message ID of the Confirmable message = True; Acknowledgement message carry a response = True; Acknowledgement message be Empty = True;")
y.append("Reset message echo the Message ID of the Confirmable message = True; Reset message be Empty = True;")
y.append(
    "recipients of Acknowledgement and Reset messages respond with either Acknowledgement or Reset messages = False;")
y.append("endpoint keep track of timeout = True; endpoint keep track of retransmission counter = True;")
y.append("the entire sequence of (re-)transmissions stay in the envelope of MAX_TRANSMIT_SPAN = True;")
y.append("endpoint sent a Confirmable message = True; endpoint give up in attempting to obtain an ACK = True;")
y.append(
    "responder rely on this cross-layer behavior from a requester = False; responder retain the state to create the ACK for the request = True;")
y.append("receipt of ICMP errors = True; give up retransmission = True;")
y.append(
    "take account of ICMP errors = True; check the original datagram in the ICMP message = True; not possible due to limitations of the UDP service API = True; ICMP errors be ignored = True;")
y.append(
    "Packet Too Big errors be ignored = True; implementation note is followed = True; Packet Too Big errors feed into a path MTU discovery algorithm = True;")
y.append("Source Quench and Time Exceeded ICMP messages be ignored = True;")
y.append(
    "errors after appropriate vetting = True; errors be used to inform the application of a failure in sending = True;")
y.append(
    "Non-confirmable message carries either a request or response = True; Non-confirmable message be Empty = False;")
y.append("Non-confirmable message be acknowledged = False;")
y.append("recipient reject the message = True; recipient lacks context to process the message properly = True;")
y.append(
    "'Rejecting a Non-confirmable message = True; sending a matching Reset message = True; rejected message be silently ignored = True;")
y.append("sender transmit multiple copies of a Non-confirmable message within MAX_TRANSMIT_SPAN = True;")
y.append("message ID be echoed in Acknowledgement or Reset messages = True;")
y.append("same Message ID be reused within the EXCHANGE_LIFETIME = False;")
y.append(
    "Acknowledgement or Reset message to match a Confirmable or Non-confirmable message = True; Message ID and source endpoint match the Message ID and destination endpoint = True;")
y.append(
    "recipient acknowledge each duplicate copy of a Confirmable message using the same Acknowledgement or Reset message = True; recipient process any request or response in the message only once = True;")
y.append(
    "rule be relaxed = True; Confirmable message transports a request that is idempotent = True; request be handled in an idempotent fashion = True;")
y.append(
    "rule be relaxed = True; recipient silently ignore any duplicated Non-confirmable messages = True; recipient process any request or response in the message only once = True;")
y.append("message fit within a single IP packet = True; message fit within a single IP datagram = True;")
y.append("Path MTU is known for a destination = False; IP MTU of 1280 bytes be assumed = True;")
y.append("client limit the number of simultaneous outstanding interactions to NSTART = True;")
y.append(
    "modified by additional congestion control optimizations = True; it be chosen in such a way that an endpoint does not exceed an average data rate of PROBING_RATE = True;")
y.append("server implement some rate limiting for its response transmissions = True;")
y.append("an application environment use consistent values for these parameters = True;")
y.append(
    "Configuration decrease ACK_TIMEOUT or increase NSTART without using mechanisms that ensure congestion control safety = False;")
y.append(
    "ACK_RANDOM_FACTOR be decreased below 1.0 = False; ACK_RANDOM_FACTOR have a value that is sufficiently different from 1.0 = True;")
y.append(
    "the choice of transmission parameters leads to an increase of derived time values = True; configuration ensure the adjusted value is also available to all the endpoints = True;")
y.append("GET take any other action on a resource other than retrieval = False;")
y.append("GET, PUT, and DELETE be performed in such a way that they are idempotent = True;")
y.append("client be prepared to receive either = True;")
y.append(
    "server send back an Empty Acknowledgement = True; server send back the response in another Acknowledgement = False;")
y.append(
    "retransmitted request is received = True; Empty Acknowledgement be sent = True; any response be sent as separate response = True;")
y.append("server sends Confirmable response = True; client Acknowledgement be Empty message = True;")
y.append("server stop retransmitting response = True; matching Acknowledgement = True; matching Reset = True;")
y.append("request message is Non-confirmable = True; response be returned in Non-confirmable message = True;")
y.append(
    "endpoint be prepared to receive a Non-confirmable response in reply to a Confirmable request = True; endpoint be prepraed to receive a Confirmable response in reply to a Non-confirmable request = True;")
y.append("server echo client-generated token in response = True;")
y.append("client generate unique tokens = True;")
y.append(
    "client send request without using Transport Layer Security = True; client use a nontrivial and randomized token = True;")
y.append("client connected to the general Internet = True; client use at least 32 bits of randomness = True;")
y.append(
    "endpoint receiving a token it did not generate = True; treat the token as opaque and make no assumptions about its content or structure = True;")
y.append("source endpoint of the response be the same as the destination endpoint of the original request = True;")
y.append(
    "Message ID of the Confirmable request and the Acknowledgement match = True; the tokens of the response and original request match = True;")
y.append("the tokens of the response and original request match = True;")
y.append(
    "option is not defined for a Method or Response Code = True; option be included by a sender = False; option be treated like an unrecognized option = True;")
y.append("unrecognized options of class elective = True; be silently ignored = True;")
y.append(
    "Unrecognized options of class critical that occur in a Confirmable request = True; return of a 4.02 (Bad Option) response = True;")
y.append("response include a diagnostic payload describing the unrecognized option = True;")
y.append(
    "Unrecognized options of class critical that occur in a Confirmable response = True; Unrecognized options of class critical that piggybacked in an Acknowledgement = True; the response to be rejected = True;")
y.append(
    "Unrecognized options of class critical that occur in a Non-confirmable message = True; the message to be rejected = True;")
y.append(
    "the length of an option value in a request is outside the defined range = True; option be treated like an unrecognized option = True;")
y.append(
    "the value of an option is intended to be this default value = True; option be included in the message = False;")
y.append("option not present = True; default value be assumed = True;")
y.append("option is repeatable = True; option be included one or more times = True;")
y.append("option is repeatable = False; option be included more than once = False;")
y.append(
    "message includes an option with more occurrences than the option is defined for = True; supernumerary option be treated like an unrecognized option = True;")
y.append(
    "Method or Response Code is defined to have a payload = False; sender include one = False; recipient ignore it = True;")
y.append("Payload sniffing be attempted = True; no content type is given = True;")
y.append("Not applicable;")
y.append("diagnostic message be encoded using UTF-8 = True;")
y.append("no additional information beyond the Response Code = True; payload be empty = True;")
y.append("Caching CoAP endpoints cache responses = True;")
y.append(
    "Response Codes indicate success and are unrecognized by an endpoint = True; Response Codes be cached = False;")
y.append("Not applicable;")
y.append("server wish to prevent caching = True; server include a Max-Age Option with a value of zero seconds = True;")
y.append("endpoint add an ETag Option = True;")
y.append("response be used to satisfy the request = True; response replace the stored response = True;")
y.append(
    "client uses a proxy to make a request that will use a secure URI scheme = True; request be sent using DTLS = True; equivalent lower-layer security is used = True;")
y.append("request to the destination times out = True; 5.04 (Gateway Timeout) response be returned = True;")
y.append(
    "request returns a response that cannot be processed by the proxy = True; 5.02 (Bad Gateway) response be returned = True;")
y.append(
    "response is generated out of a cache = True; the generated Max-Age Option extend the max-age originally set by the server = False;")
y.append("options present in a proxy request = True; options be processed at the proxy = True;")
y.append(
    "Unsafe options in a request that are not recognized by the proxy = True; 4.02 (Bad Option) response be returned = True;")
y.append(
    "CoAP-to-CoAP proxy forward to the origin server all Safe-to-Forward options that it does not recognize = True;")
y.append("Unsafe options in a response that are not recognized = True; 5.02 (Bad Gateway) response be returned = True;")
y.append("Safe-to-Forward options not recognized = True; Safe-to-Forward options be forwarded = True;")
y.append(
    "endpoint is unwilling or unable to act as proxy for the request URI = True; 5.05 (Proxying Not Supported) response be returned = True;")
y.append(
    "the authority is recognized as identifying the proxy endpoint itself = True; request be treated as a local request = True;")
y.append(
    "proxy is configured to forward the proxy request to another proxy = True; the scheme of the request URI defines the outgoing protocol and its details = True;")
y.append(
    "unrecognized or unsupported Method Code = True; 4.05 (Method Not Allowed) piggybacked response be returned = True;")
y.append("success = True; 2.05 (Content) or 2.03 (Valid) Response Code be returned = True;")
# Annotate another 100
y.append(
    "resource be created on server = True; 2.01 (Created) Response Code be returned = True; response include the URI of the new resource = True;")
y.append(
    "POST succeeds = True; new resource being created on the server = False; 2.04 (Changed) Response Code be returned = True;")
y.append("POST succeeds = True; target resource being deleted = True; 2.02 (Deleted) Response Code be returned = True;")
y.append(
    "resource exists at the request URI = True; enclosed representation be considered a modified version of resource = True; 2.04 (Changed) Response Code be returned = True;")
y.append(
    "no resource exists = True; server create a new resource = True; 2.01 (Created) Response Code be returned = True;")
y.append("resource could not be created or modified = True; error Response Code be sent = True;")
y.append("2.02 (Deleted) Response Code be used = True; success = True; resource exist before the request = False;")
y.append(
    "cache receiving this response = True; cachemark any stored response for the created resource as not fresh = True;")
y.append("cache mark any stored response for the deleted resource as not fresh = True;")
y.append("response include an ETag Option = True; response include a payload = True;")
y.append(
    "cache recognizes and processes the ETag response option = True; receives a 2.03 (Valid) response = True; cache update the stored response with the value of the Max-Age Option included in the response = True;")
y.append(
    "Safe-to-Forward option present in the stored response = True; Safe-to-Forward option be replaced with the set of options of this type in the response received = True;")
y.append("cache mark any stored response for the changed resource as not fresh = True;")
y.append("server include a diagnostic payload = True;")
y.append("client repeat the request without first improving its authentication status to the server = False;")
y.append("client repeat the request without modification = False;")
y.append(
    "response include a Size1 Option = True; server is not in a position to make this information available = True;")
y.append("server include a diagnostic payload = True;")
y.append("value of a Uri-Path Option be '.' = False;")
y.append(
    "forward-proxy forward the request on to another proxy = True; forward-proxy forward the request directly to the server specified by the absolute-URI = True;")
y.append("proxy be able to recognize all of its server names = True;")
y.append(
    "endpoint receiving a request with a Proxy-Uri Option that is unable or unwilling to act as a forward-proxy for the request = True; 5.05 (Proxying Not Supported) response be returned = True;")
y.append(
    "Proxy-Uri Option take precedence over any of the Uri-Host, Uri-Port, Uri-Path or Uri-Query options = True; Uri-Host, Uri-Port, Uri-Path or Uri-Query options be included in a request containing the Proxy-Uri Option = False;")
y.append(
    "preferred Content-Format cannot be returned = True; 4.06 (Not Acceptable) response be returned = True; another error code takes precedence for this response = True;")
y.append(
    "Servers that provide resources with strict tolerances on the value of Max-Age = True; server update the value before each retransmission = True;")
y.append(
    "endpoint receiving an entity-tag = True; endpoint treat it as opaque and make no assumptions about its content or structure = True;")
y.append("ETag Option occur more than once in a response = False;")
y.append("ETag Option occur zero, one, or multiple times in a request = True;")
y.append(
    "response with one or more Location-Path and/or Location-Query Options passes through a cache that interprets these options = True; implied URI identifies one or more currently stored responses = True; entries be marked as not fresh = True;")
y.append("value of a Location-Path Option be '.' = False;")
y.append(
    "reserved option numbers occurs in addition to Location-Path and/or Location-Query and are not supported = True; 4.02 (Bad Option) response be returned = True;")
y.append("condition given is fulfilled = False; server perform the requested method = False;")
y.append("server respond with the 4.12 (Precondition Failed) Response Code = True;")
y.append(
    "request result in anything other than a 2.xx or 4.12 Response Code without the conditional request options = True; any conditional request options be ignored = True;")
y.append("If-Match Option be used to make a request conditional on the current existence or value of an ETag = True;")
y.append("If-None-Match Option be used to make a request conditional on the nonexistence of the target resource = True;")
y.append("host be empty = False; URI is received with a missing authority or an empty host = True; URI be considered invalid = True;")
y.append("UDP datagrams be secured through the use of DTLS = True;")
y.append("the hexadecimal notation for percent-encoding in CoAP URIs use uppercase letters = True;")
y.append("CoAP default port number 5683 be supported by a server that offers resources for resource discovery = True; CoAP default port number 5683 be supported for providing access to other resources = True;")
y.append("default port number 5684 for DTLS-secured CoAP be supported by a server for resource discovery and for providing access to other resources = True;")
y.append("CoAP endpoint support the CoRE Link Format of discoverable resources = True; fully manual configuration is desired = True;")
y.append("value be in the range of 0-65535 (16-bit unsigned integer) = True;")
y.append("Content-Format code include a space-separated sequence of Content-Format codes = True;")
y.append("endpoint be prepared to receive such messages = True; endpoint ignore them = True; multicast service discovery is not desired = True;")
y.append("multicast requests be Non-confirmable = True;")
y.append("server be aware that a request arrived via multicast = True;")
y.append("server is aware that a request arrived via multicast = True; return a Reset message in reply to a Non-confirmable message = False;")
y.append("it is aware = False; return a Reset message in reply to a Non-confirmable message = True;")
y.append("sender using a Message ID that is also still active = False;")
y.append("server is aware that a request arrived via multicast = True; server ignore the request = True;")
y.append("specific value of this Leisure be derived = True;")
y.append("server pick a random point of time within the chosen leisure period to send back the unicast response to the multicast request = True;")
y.append("CoAP endpoint does not have suitable data to compute a value for Leisure = True; CoAP endpoint resort to DEFAULT_LEISURE = True;")
y.append("matching a response to a multicast request = True; token match = True;")
y.append("It update a cache with the received responses = True;")
y.append("response received in reply to a GET request to a multicast group = True; response be used to satisfy a subsequent request on the related unicast request URI = True;")
y.append("cache revalidate a response by making a GET request on the related unicast request URI = True;")
y.append("GET request to a multicast group contain an ETag option = False;")
y.append("Alternative techniques to provide lower-layer security be used when appropriate = True;")
y.append("CoAP messages be sent as DTLS 'application data' = True;")
y.append("matching an Acknowledgement message or Reset message to a Confirmable message = True; matching a Reset message to a Non-confirmable message = True; DTLS session be the same = True; epoch be the same = True;")
y.append("Retransmissions be performed across epochs = False;")
y.append("matching a response to a request = True; DTLS session be the same = True; epoch be the same = True;")
y.append("response to a DTLS secured request always be DTLS secured using the same security session and epoch = True;")
y.append("attempt to supply a NoSec response to a DTLS request = False; response be rejected = True; match an unrelated NoSec request = True;")
y.append("Endpoint Identity Devices support the Server Name Indication (SNI) = True;")
y.append("Implementations in these modes support the mandatory-to-implement cipher suite TLS_PSK_WITH_AES_128_CCM_8 = True;")
y.append("device be configured with multiple raw public keys;")
y.append("Implementations in RawPublicKey mode support the mandatory-to-implement cipher suite TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8 = True;")
y.append("key used be ECDSA capable = True;")
y.append("curve secp256r1 be supported = True;")
y.append("Implementations use the Supported Elliptic Curves = True; Implementations use the Supported Point Formats Extensions = True; uncompressed point format be supported = True;")
y.append("implementations support checking RawPublicKey identities = True; implementations support at least the sha-256-120 mode = True;")
y.append("Implementations support longer length identifiers = True; Implementations support shorter lengths = True;")
y.append("their use = False;")
y.append("implementations support the binary mode = True; implementations have a user interface = True; implementations support the human-speakable format = True;")
y.append("access control list of identifiers with which the device may start DTLS sessions be installed and maintained = True;")
y.append("Implementations in Certificate Mode support the mandatory-to-implement cipher suite TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8 = True;")
y.append("Certificates be signed ECDSA using secp256r1 = True; signature use SHA-256 = True;")
y.append("key used be ECDSA capable = True;")
y.append("curve secp256r1 MUST be supported = True;")
y.append("Implementations use the Supported Elliptic Curves = True; Implementations use the Supported Point Formats Extensions = True; uncompressed point format be supported = True;")
y.append("CoAP node has a source of absolute time = True; node check that the validity dates of the certificate are within range = True;")
y.append("certificate be validated as appropriate for the security requirements = True;")
y.append("certificate contains a SubjectAltName = True; the authority of the request URI match at least one of the authorities of any CoAP URI found in a field of URI type in the SubjectAltName set = True;")
y.append("there is SubjectAltName in the certificate = False; the authority of the request URI match the Common Name (CN) found in the certificate = True;")
y.append("system has a shared key in addition to the certificate = True; cipher suite that includes the shared key be used = True;")
y.append("underlying model of Confirmable or Non-confirmable messages have no effect on a proxy function = True;")
y.append("success = True; 2.05 (Content) Response Code be returned = True;")
y.append("payload of the response be a representation of the target HTTP resource = True; Content-Format Option be set accordingly = True;")
y.append("response indicate a Max-Age value that is no greater than the remaining time the representation can be considered fresh = True;")
y.append("HTTP entity has an entity-tag = True; proxy include an ETag Option in the response = True; proxy process the ETag Option in the request = True;")
y.append("GET request include an Accept Option = True;")
y.append("request include one or more ETag Options = True;")
y.append("new resource is created at the request URI = True; 2.01 (Created) Response Code be returned = True;")
y.append("existing resource is modified = True; 2.04 (Changed) Response Code be returned = True;")
y.append("2.02 (Deleted) Response Code be returned = True; success = True; resource does not exist at the time of the request = True;")
y.append("the action performed by the POST method does not result in a resource that can be identified by a URI = True; 2.04 (Changed) Response Code be returned = True;")
y.append("resource has been created on the origin server = True; 2.01 (Created) Response Code be returned = True;")
y.append("Not applicable = True;")
y.append("OPTIONS and TRACE methods are not supported in CoAP = True; 501 (Not Implemented) error code be returned = True;")
y.append("payload of the response be a representation of the target CoAP resource = True; Content-Type and Content-Encoding header fields be set accordingly = True;")
y.append("response indicate a max-age directive that indicates a value no greater than the remaining time the representation can be considered fresh = True;")
y.append("proxy retry the request with further media types from the HTTP Accept header field = True;")
y.append("server return a message-body in the response = False;")
y.append("action performed by the POST method does not result in a resource that can be identified by a URI = True; 200 (OK) Response Code be returned = True; 204 (No Content) Response Code be returned = True;")
y.append("resource has been created on the origin server = True; 201 (Created) Response Code be returned = True;")
y.append("caching proxy make cached values available to requests that have lesser transport-security properties = False;")
y.append("responses to 'coaps' identified requests are never 'public' = True; responses be reused for shared caching = False; the cache is able to make equivalent access control decisions to the ones that led to the cached entry = True;")
y.append("large amplification factors be provided in the response = False; the request is authenticated = False;")
y.append("CoAP servers accept multicast requests that can not be authenticated = False;")
y.append("possible = True; CoAP server limit the support for multicast requests = True;")
y.append("Implementations make use of modern APIs = True; available = True;")
y.append("that is the case = True; node be used for processes that require good entropy = False;")
y.append("They are be used in operational deployments = False;")
y.append("They are be used in operational deployments = False;")

train_data = []
for i in range(len(y)):
    train_data.append(f"<|startoftext|>Sentence: {rule_sentences[i]}\nBehaviours: {y[i]} <|endoftext|>")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer(train_data, return_tensors="pt", padding=True, truncation=True)
labels = inputs["input_ids"].clone()
inputs["labels"] = labels

train_dataset = MeditationDataset(inputs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(200):
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        model.train()
        train_loss = train(train_batch, model, optimizer)
        train_loop.set_postfix(train_loss=train_loss.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Epoch {epoch} train")
    average_train_loss = overall_train_loss / num_of_train_batches
    print(f"average train loss: {average_train_loss}")

    if average_train_loss < 0.05:
        break

torch.save(model, r"../model/variable_extractor_GPT-2.pt")
