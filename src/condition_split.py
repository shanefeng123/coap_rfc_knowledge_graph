from transformers import PegasusForConditionalGeneration, PegasusTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AdamW
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
        decoded_train[i] = decoded_train[i].split("\n")[0].split("Sentence:")[1].strip()

    for sentence in decoded_train:
        train_generated.append(tokenizer.decode(generate(sentence, model, tokenizer)[0], skip_special_tokens=True))
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
        decoded_test[i] = decoded_test[i].split("\n")[0].split("Sentence:")[1].strip()

    for sentence in decoded_test:
        test_generated.append(tokenizer.decode(generate(sentence, model, tokenizer)[0], skip_special_tokens=True))
    return test_loss, test_decoded_labels, test_generated


def generate(sentence, model, tokenizer):
    data = f"<|startoftext|>Sentence: {sentence}\nAntecedent:"
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024, pad_token_id=tokenizer.eos_token_id)
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
y.append(
    "Antecedent: If the value of an option is intended to be this default value;\nConsequent: the option SHOULD NOT be included in the message;")
y.append("Antecedent: If the option is not present;\nConsequent: the default value MUST be assumed;")
y.append("Antecedent: An option that is repeatable;\nConsequent: MAY be included one or more times in a message;")
y.append("Antecedent: An option that is not repeatable;\nConsequent: MUST NOT be included more than once in a message;")
y.append(
    "Antecedent: If a message includes an option with more occurrences than the option is defined for;\nConsequent: each supernumerary option occurrence that appears subsequently in the message MUST be treated like an unrecognized option (see Section 5.4.1);")
y.append(
    "Antecedent: If a Method or Response Code is not defined to have a payload;\nConsequent: then a sender MUST NOT include one, and a recipient MUST ignore it;")
y.append('Antecedent: if no content type is given;\nConsequent: Payload "sniffing" SHOULD only be attempted;')
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: if there is no additional information beyond the Response Code;\nConsequent: In contrast to what is usual in HTTP, the payload SHOULD be empty;")
# 80
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: Response Codes that indicate success and are unrecognized by an endpoint;\nConsequent: MUST NOT be cached;")
y.append(
    "Antecedent: unless: o the presented request method and that used to obtain the stored response match, o all options match between those in the presented request and those of the request used to obtain the stored response (which includes the request URI), except that there is no need for a match of any request options marked as NoCacheKey (Section 5.4) or recognized by the Cache and fully interpreted with respect to its specified cache behavior (such as the ETag request option described in Section 5.10.6; see also Section 5.4.2), and o the stored response is either fresh or successfully validated as defined below;\nConsequent: For a presented request, a CoAP endpoint MUST NOT use a stored response;")
y.append(
    "Antecedent: If an origin server wishes to prevent caching;\nConsequent: it MUST explicitly include a Max-Age Option with a value of zero seconds;")
y.append(
    "Antecedent: When sending such a request;\nConsequent: the endpoint SHOULD add an ETag Option specifying the entity-tag of each stored response that is applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    'Antecedent: When a client uses a proxy to make a request that will use a secure URI scheme (e.g., "coaps" or "https");\nConsequent: the request towards the proxy SHOULD be sent using DTLS except where equivalent lower-layer security is used for the leg between the client and the proxy;')
y.append(
    "Antecedent: If the request to the destination times out;\nConsequent: then a 5.04 (Gateway Timeout) response MUST be returned;")
y.append(
    "Antecedent: If the request to the destination returns a response that cannot be processed by the proxy (e.g, due to unrecognized critical options or message format errors);\nConsequent: then a 5.02 (Bad Gateway) response MUST be returned;")
y.append(
    "Antecedent: If a response is generated out of a cache;\nConsequent: the generated (or implied) Max-Age Option MUST NOT extend the max-age originally set by the server, considering the time the resource representation spent in the cache;")
# 90
y.append("Antecedent: All options present in a proxy request;\nConsequent: MUST be processed at the proxy;")
y.append(
    "Antecedent: Unsafe options in a request that are not recognized by the proxy;\nConsequent: MUST lead to a 4.02 (Bad Option) response being returned by the proxy;")
y.append(
    "Antecedent: all Safe-to-Forward options that it does not recognize;\nConsequent: A CoAP-to-CoAP proxy MUST forward to the origin server all Safe-to-Forward options;")
y.append(
    "Antecedent: Similarly, Unsafe options in a response that are not recognized by the CoAP-to-CoAP proxy server;\nConsequent: MUST lead to a 5.02 (Bad Gateway) response;")
y.append("Antecedent: Again, Safe-to-Forward options that are not recognized;\nConsequent: MUST be forwarded;")
y.append(
    "Antecedent: When a proxy request is made to an endpoint and the endpoint is unwilling or unable to act as proxy for the request URI;\nConsequent: it MUST return a 5.05 (Proxying Not Supported) response;")
y.append(
    "Antecedent: If the authority (host and port) is recognized as identifying the proxy endpoint itself (see Section 5.10.2);\nConsequent: then the request MUST be treated as a local (non-proxied) request;")
y.append(
    'Antecedent: Unless a proxy is configured to forward the proxy request to another proxy;\nConsequent: it MUST translate the request as follows: the scheme of the request URI defines the outgoing protocol and its details (e.g., CoAP is used over UDP for the "coap" scheme and over DTLS for the "coaps" scheme);')
y.append(
    "Antecedent: A request with an unrecognized or unsupported Method Code;\nConsequent: MUST generate a 4.05 (Method Not Allowed) piggybacked response;")
y.append(
    "Antecedent: Upon success;\nConsequent: a 2.05 (Content) or 2.03 (Valid) Response Code SHOULD be present in the response;")
# 100
y.append(
    "Antecedent: If a resource has been created on the server;\nConsequent: the response returned by the server SHOULD have a 2.01 (Created) Response Code and SHOULD include the URI of the new resource in a sequence of one or more Location-Path and/or Location-Query Options (Section 5.10.7);")
y.append(
    "Antecedent: If the POST succeeds but does not result in a new resource being created on the server;\nConsequent: the response SHOULD have a 2.04 (Changed) Response Code;")
y.append(
    "Antecedent: If the POST succeeds and results in the target resource being deleted;\nConsequent: the response SHOULD have a 2.02 (Deleted) Response Code;")
y.append(
    "Antecedent: If a resource exists at the request URI;\nConsequent:  the enclosed representation SHOULD be considered a modified version of that resource, and a 2.04 (Changed) Response Code SHOULD be returned;")
y.append(
    "Antecedent: If no resource exists;\nConsequent: then the server MAY create a new resource with that URI, resulting in a 2.01 (Created) Response Code;")
y.append(
    "Antecedent: If the resource could not be created or modified;\nConsequent: then an appropriate error Response Code SHOULD be sent;")
y.append(
    "Antecedent: on success or in case the resource did not exist before the request;\nConsequent: A 2.02 (Deleted) Response Code SHOULD be used;")
y.append(
    "Antecedent: A cache receiving this response;\nConsequent: MUST mark any stored response for the created resource as not fresh;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 110
y.append(
    "Antecedent: When a cache that recognizes and processes the ETag response option receives a 2.03 (Valid) response;\nConsequent: it MUST update the stored response with the value of the Max-Age Option included in the response (explicitly, or implicitly as a default value; see also Section 5.6.2);")
y.append(
    "Antecedent: For each type of Safe-to-Forward option present in the response, the (possibly empty) set of options of this type that are present in the stored response;\nConsequent: MUST be replaced with the set of options of this type in the response received;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: without first improving its authentication status to the server；\nConsequent: The client SHOULD NOT repeat the request;")
y.append("Antecedent: without modification;\nConsequent: The client SHOULD NOT repeat the request;")
y.append(
    "Antecedent: unless the server is not in a position to make this information available;\nConsequent: The response SHOULD include a Size1 Option (Section 5.10.9) to indicate the maximum size of request entity the server is able and willing to handle;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 120
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: An endpoint receiving a request with a Proxy-Uri Option that is unable or unwilling to act as a forward-proxy for the request;\nConsequent: MUST cause the return of a 5.05 (Proxying Not Supported) response;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    'Antecedent: If the preferred Content-Format cannot be returned, unless another error code takes precedence for this response;\nConsequent: then a 4.06 "Not Acceptable" MUST be sent as a response;')
y.append(
    "Antecedent: Servers that provide resources with strict tolerances on the value of Max-Age;\nConsequent: SHOULD update the value before each retransmission;")
y.append(
    "Antecedent: An endpoint receiving an entity-tag;\nConsequent: MUST treat it as opaque and make no assumptions about its content or structure;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If a response with one or more Location-Path and/or Location-Query Options passes through a cache that interprets these options and the implied URI identifies one or more currently stored responses;\nConsequent: those entries MUST be marked as not fresh;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 130
y.append(
    "Antecedent: If any of these reserved option numbers occurs in addition to Location-Path and/or Location-Query and are not supported;\nConsequent: then a 4.02 (Bad Option) error MUST be returned;")
y.append(
    "Antecedent: For each of these options, if the condition given is not fulfilled;\nConsequent: then the server MUST NOT perform the requested method;")
y.append("Antecedent: instead;\nConsequent: the server MUST respond with the 4.12 (Precondition Failed) Response Code;")
y.append(
    "Antecedent: If the request would, without the conditional request options, result in anything other than a 2.xx or 4.12 Response Code;\nConsequent: then any conditional request options MAY be ignored;")
y.append(
    "Antecedent: conditional on the current existence or value of an ETag for one or more representations of the target resource;\nConsequent: The If-Match Option MAY be used to make a request;")
y.append(
    "Antecedent: conditional on the nonexistence of the target resource;\nConsequent: The If-None-Match Option MAY be used to make a request;")
y.append(
    "Antecedent: if a URI is received with a missing authority or an empty host;\nConsequent: then it MUST be considered invalid;")
y.append(
    "Antecedent: if the port subcomponent is empty or not given;\nConsequent: a default UDP port of 5684 is assumed; and the UDP datagrams MUST be secured through the use of DTLS as described in Section 9.1;")
y.append(
    "Antecedent: Not applicable;\nConsequent: the server MUST respond with the 4.12 (Precondition Failed) Response Code;")
y.append(
    "Antecedent: server that offers resources for resource discovery;\nConsequent: The CoAP default port number 5683 MUST be supported by a server and SHOULD be supported for providing access to other resources;")
# 140
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: except where fully manual configuration is desired;\nConsequent: To maximize interoperability in a CoRE environment, a CoAP endpoint SHOULD support the CoRE Link Format of discoverable resources as described in [RFC6690];")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: if multicast service discovery is not desired;\nConsequent: Note that an endpoint might receive multicast requests on other multicast addresses, including the all-nodes IPv6 address (or via broadcast on IPv4); an endpoint MUST therefore be prepared to receive such messages but MAY ignore them;")
y.append("Antecedent: Such multicast requests;\nConsequent: MUST be Non-confirmable;")
y.append(
    "Antecedent: if available;\nConsequent: A server SHOULD be aware that a request arrived via multicast, e.g., by making use of modern APIs such as IPV6_RECVPKTINFO [RFC3542];")
y.append(
    "Antecedent: To avoid an implosion of error responses, when a server is aware that a request arrived via multicast;\nConsequent: it MUST NOT return a Reset message in reply to a Non-confirmable message;")
y.append(
    "Antecedent: If it is not aware;\nConsequent: it MAY return a Reset message in reply to a Non-confirmable message as usual;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 150
y.append(
    "Antecedent: When a server is aware that a request arrived via multicast, in particular if it doesn\'t have anything useful to respond (e.g., if it only has an empty payload or an error response);\nConsequent: the server MAY always ignore the request;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If a CoAP endpoint does not have suitable data to compute a value for Leisure;\nConsequent: it MAY resort to DEFAULT_LEISURE;")
y.append(
    "Antecedent: When matching a response to a multicast request;\nConsequent:  only the token MUST match; the source endpoint of the response does not need to (and will not) be the same as the destination endpoint of the original request;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: A response received in reply to a GET request to a multicast group;\nConsequent: MAY be used to satisfy a subsequent request on the related unicast request URI;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: A GET request to a multicast group;\nConsequent: MUST NOT contain an ETag option;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 160
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: The following rules are added for matching an Acknowledgement message or Reset message to a Confirmable message, or a Reset message to a Non-confirmable message;\nConsequent: The DTLS session MUST be the same, and the epoch MUST be the same;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: This means the response to a DTLS secured request;\nConsequent: MUST always be DTLS secured using the same security session and epoch;")
y.append(
    "Antecedent: Any attempt to supply a NoSec response to a DTLS request simply does not match the request (unless it does match an unrelated NoSec request);\nConsequent: and therefore MUST be rejected;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 170
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: All implementations that support checking RawPublicKey identities;\nConsequent: MUST support at least the sha-256-120 mode (SHA-256 truncated to 120 bits);")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: implementations that have a user interface;\nConsequent: All implementations SHOULD support the binary mode, SHOULD also support the human-speakable format;")
y.append(
    "Antecedent: During (initial and ongoing) provisioning;\nConsequent: an access control list of identifiers with which the device may start DTLS sessions SHOULD also be installed and maintained;")
y.append(
    "Antecedent: X.509 Certificates Implementations in Certificate Mode;\nConsequent: MUST support the mandatory-to-implement cipher suite TLS_ECDHE_ECDSA_WITH_AES_128_CCM_8 as specified in [RFC7251], [RFC5246], and [RFC4492];")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
# 180
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the CoAP node has a source of absolute time;\nConsequent: then the node SHOULD check that the validity dates of the certificate are within range;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the certificate contains a SubjectAltName;\nConsequent: then the authority of the request URI MUST match at least one of the authorities of any CoAP URI found in a field of URI type in the SubjectAltName set;")
y.append(
    "Antecedent: If there is no SubjectAltName in the certificate;\nConsequent: then the authority of the request URI MUST match the Common Name (CN) found in the certificate using the matching rules defined in [RFC3280] with the exception that certificates with wildcards are not allowed;")
y.append(
    "Antecedent: If the system has a shared key in addition to the certificate;\nConsequent: then a cipher suite that includes the shared key such as TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA [RFC5489] SHOULD be used;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Upon success;\nConsequent: a 2.05 (Content) Response Code SHOULD be returned;")
# 190
y.append(
    "Antecedent: The payload of the response;\nConsequent: MUST be a representation of the target HTTP resource, and the Content-Format Option MUST be set accordingly;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the HTTP entity has an entity-tag;\nConsequent: the proxy SHOULD include an ETag Option in the response and process ETag Options in requests as described below;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If a new resource is created at the request URI;\nConsequent: a 2.01 (Created) response MUST be returned to the client;")
y.append(
    "Antecedent: If an existing resource is modified;\nConsequent: a 2.04 (Changed) response MUST be returned to indicate successful completion of the request;")
y.append(
    "Antecedent: upon success or if the resource does not exist at the time of the request;\nConsequent: A 2.02 (Deleted) response MUST be returned to the client;")
y.append(
    "Antecedent: If the action performed by the POST method does not result in a resource that can be identified by a URI;\nConsequent: a 2.04 (Changed) response MUST be returned to the client;")
y.append(
    "Antecedent: If a resource has been created on the origin server;\nConsequent: a 2.01 (Created) response MUST be returned;")
# 200
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: As the OPTIONS and TRACE methods are not supported in CoAP;\nConsequent: a 501 (Not Implemented) error MUST be returned to the client;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    "Antecedent: If the action performed by the POST method does not result in a resource that can be identified by a URI;\nConsequent: a 200 (OK) or 204 (No Content) response MUST be returned to the client;")
y.append(
    "Antecedent: If a resource has been created on the origin server;\nConsequent: a 201 (Created) response MUST be returned;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append(
    'Antecedent: unless the cache is able to make equivalent access control decisions to the ones that led to the cached entry;\nConsequent: Unlike the "coap" scheme, responses to "coaps" identified requests are never "public" and thus MUST NOT be reused for shared caching;')
# 210
y.append(
    "Antecedent: if the request is not authenticated;\nConsequent: Therefore, large amplification factors SHOULD NOT be provided in the response;")
y.append(
    "Antecedent: multicast requests that can not be authenticated in some way, cryptographically or by some multicast boundary limiting the potential sources;\nConsequent: To limit the possibility of malicious use, CoAP servers SHOULD NOT accept multicast requests;")
y.append(
    "Antecedent: If possible;\nConsequent: a CoAP server SHOULD limit the support for multicast requests to the specific resources where the feature is required;")
y.append(
    "Antecedent: if available;\nConsequent: Implementations SHOULD make use of modern APIs such as IPV6_RECVPKTINFO [RFC3542] to make this determination;")
y.append(
    "Antecedent: If that is the case;\nConsequent: the node MUST NOT be used for processes that require good entropy, such as key generation;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")
y.append("Antecedent: Not applicable;\nConsequent: Not applicable;")

for i in range(len(y)):
    if not y[i].startswith("Antecedent:"):
        print(i)

print(len(y))
# data = []
# for i in range(len(y)):
#     data.append(f"<|startoftext|>Sentence: {rfc7252_rule_sentences[i]}\n{y[i]}<|endoftext|>")
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
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)
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
#
#         for i in range(len(train_decoded_labels)):
#             train_decoded_labels[i] = train_decoded_labels[i].split("Antecedent:", 1)[1].strip()
#             train_generated[i] = train_generated[i].split("Antecedent:", 1)[1].strip()
#
#         for i in range(len(train_decoded_labels)):
#             train_batch_bleu += sentence_bleu([train_decoded_labels[i]], train_generated[i])
#         train_loop.set_postfix(train_loss=train_loss.item())
#         overall_train_loss += train_loss.item()
#         train_loop.set_description(f"Epoch {epoch} train")
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
#         for i in range(len(test_decoded_labels)):
#             test_decoded_labels[i] = test_decoded_labels[i].split("Antecedent:", 1)[1].strip()
#             test_generated[i] = test_generated[i].split("Antecedent:", 1)[1].strip()
#             test_batch_bleu += sentence_bleu([test_decoded_labels[i]], test_generated[i])
#         test_loop.set_postfix(test_loss=test_loss.item())
#         overall_test_loss += test_loss.item()
#         test_loop.set_description(f"Epoch {epoch} test")
#         overall_test_bleu += test_batch_bleu / len(test_decoded_labels)
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
#     if average_train_loss < 0.1:
#         break
#
#     with open(r"../results/condition_split_results.txt", "a") as file:
#         file.write(
#             f"Epoch {epoch} average_train_loss: {average_train_loss} average train bleu: {average_train_bleu}")
#         file.write("\n")
#         file.write(
#             f"Epoch {epoch} average_test_loss: {average_test_loss} average test bleu: {average_test_bleu}")
#         file.write("\n")
# #
# with open("../data/coap_sentence_condition_split", "wb") as file:
#     pickle.dump(y, file)
#
# torch.save(model, r"../model/condition_splitter.pt")

# for i in range(200, 217):
#     print(tokenizer.decode(generate(rfc7252_rule_sentences[i], model, tokenizer)[0]))

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
