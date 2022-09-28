from prepare_pretrain_data import prepare_pretrain_data
import os
import openai
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]

rule_sentences = []
for sentence in rfc7252:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rule_sentences.append(sentence)
            break


def construct_contextual_prompt(query_sentence_id, context_sentence_ids):
    """
    Construct a contextual prompt for OpenAI's GPT-3 API.
    Args:
        query_sentence_id: The query sentence id in the rule sentence list
        context_sentence_ids: The context sentence ids in the rule sentence list

    Returns:
        The prompt to be sent to OpenAI's GPT-3 API

    """
    query_sentence = rule_sentences[query_sentence_id]
    context_sentences = [rule_sentences[i] for i in context_sentence_ids]
    context_behaviours = [y[i] for i in context_sentence_ids]
    prompt = ""
    for i in range(len(context_sentences)):
        prompt += "Sentence: " + context_sentences[i] + "\n"
        prompt += "Behaviours: " + context_behaviours[i] + "\n"
    prompt += "Sentence: " + query_sentence + "\n"
    prompt += "Behaviours: "
    return prompt


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


rule_sentences = rule_sentences[1:]
# Annotate 50 samples
y = []
y.append("set this field to 1 = True;")
y.append("unknown version numbers = True; be silently ignored = True;")
y.append("Lengths 9-15 = True; be sent = False; be processed as a message format error = True;")
y.append(
    "presence of a marker followed by a zero-length payload = True; be processed as a message format error = True;")
y.append("appear in order of their Option Numbers = True; delta encoding is used = True;")
y.append(
    "the field is set to this value = True; entire byte is the payload marker = False; be processed as a message format error = True;")
y.append("the field is set to this value = True; be processed as a message format error = True;")
y.append("length and format define variable-length values = True;")
y.append("options defined in other documents make use of other option value formats = True;")
y.append("has a choice = True; represent the integer with as few bytes as possible = True;")
y.append("be prepared to process values with leading zero bytes = True;")
y.append("be set to 0 = True; be present after the Message ID field = False;")
y.append("any bytes = True; be processed as a message format error = True;")
y.append(
    "acknowledge a Confirmable message with an Acknowledgement message = True; reject the message = True; lacks context to process the message properly = True;")
y.append("echo the Message ID of the Confirmable message = True; carry a response = True; be Empty = True;")
y.append("echo the Message ID of the Confirmable message = True; be Empty = True;")
y.append(
    "recipients of Acknowledgement and Reset messages = True; respond with either Acknowledgement or Reset messages = False;")
y.append("keep track of timeout = True; keep track of retransmission counter = True;")
y.append("the entire sequence of (re-)transmissions stay in the envelope of MAX_TRANSMIT_SPAN = True;")
y.append("sent a Confirmable message = True; give up in attempting to obtain an ACK = True;")
y.append(
    "rely on this cross-layer behavior from a requester = False; retain the state to create the ACK for the request = True;")
y.append("receipt of ICMP errors = True; give up retransmission = True;")
y.append(
    "take account of ICMP errors = True; check the original datagram in the ICMP message = True; not possible due to limitations of the UDP service API = True; ICMP errors be ignored = True;")
y.append(
    "Packet Too Big errors be ignored = True; implementation note is followed = True; feed into a path MTU discovery algorithm = True;")
y.append("Source Quench and Time Exceeded ICMP messages be ignored = True;")
y.append("appropriate vetting = True; errors be used to inform the application of a failure in sending = True;")
y.append("carries either a request or response = True; be Empty = False;")
y.append("be acknowledged = False;")
y.append("reject the message = True; lacks context to process the message properly = True;")
y.append(
    "'Rejecting a Non-confirmable message = True; sending a matching Reset message = True; be silently ignored = True;")
y.append("transmit multiple copies of a Non-confirmable message within MAX_TRANSMIT_SPAN = True;")
y.append("message ID be echoed in Acknowledgement or Reset messages = True;")
y.append("same Message ID be reused within the EXCHANGE_LIFETIME = False;")
y.append("Message ID and source endpoint match the Message ID and destination endpoint = True")
y.append(
    "acknowledge each duplicate copy of a Confirmable message using the same Acknowledgement or Reset message = True; process any request or response in the message only once = True;")
y.append(
    "be relaxed = True; Confirmable message transports a request that is idempotent = True; be handled in an idempotent fashion = True;")
y.append(
    "be relaxed = True; silently ignore any duplicated Non-confirmable messages = True; process any request or response in the message only once = True;")
y.append("fit within a single IP packet = True; fit within a single IP datagram = True;")
y.append("Path MTU is known for a destination = False; IP MTU of 1280 bytes be assumed = True;")
y.append("limit the number of simultaneous outstanding interactions to NSTART = True;")
y.append("be chosen in such a way that an endpoint does not exceed an average data rate of PROBING_RATE = True;")
y.append("implement some rate limiting for its response transmissions = True;")
y.append("an application environment use consistent values for these parameters = True;")
y.append(
    "decrease ACK_TIMEOUT or increase NSTART without using mechanisms that ensure congestion control safety = False;")
y.append(
    "ACK_RANDOM_FACTOR be decreased below 1.0 = False; have a value that is sufficiently different from 1.0 = True;")
y.append(
    "the choice of transmission parameters leads to an increase of derived time values = True; ensure the adjusted value is also available to all the endpoints")
y.append("take any other action on a resource other than retrieval = False;")
y.append("be performed in such a way that they are idempotent = True;")
y.append("be prepared to receive either = True;")
y.append("sends back an Empty Acknowledgement = True; send back the response in another Acknowledgement = False;")

# To better select the context sentences, we use our pretrained IoT BERT to do sentence encoding for each of the rule
# sentences, then we do a k-nn search to find the k nearest context sentences for each rule sentence.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

IoTBERT = BertModel.from_pretrained("../model/iot_bert")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer(rule_sentences, padding=True, truncation=True, return_tensors="pt")

IoTBERT.to(device)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
token_type_ids = inputs["token_type_ids"].to(device)

outputs = IoTBERT(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True)
rule_sentence_pool_embeddings = outputs[1].detach().cpu().numpy()

# Calculate the cosine similarity for each not annotated rule sentence and annotated rule sentence pair
cosine_similarities = []

for i in range(len(y), len(rule_sentences)):
    similarities = []
    for j in range(len(y)):
        similarities.append((cosine_similarity(rule_sentence_pool_embeddings[i], rule_sentence_pool_embeddings[j]), j))
    cosine_similarities.append(similarities)

for similarities in cosine_similarities:
    similarities.sort(reverse=True)

# Select the top k context sentences for each rule sentence, construct the prompt and call the GPT-3 API to generate the
# desired behaviour variables
# openai.api_key = "YOUR_API_KEY"
# k = 5
# extracted_behaviour_variables = []
# for i in range(len(y), len(rule_sentences)):
#     j = 0
#     context_sentence_ids = [sentence[1] for sentence in cosine_similarities[j][:k]]
#     prompt = construct_contextual_prompt(i, context_sentence_ids)
#     extracted_behaviour_variables.append(openai.Completion.create(
#         model="text-davinci-002",
#         prompt=prompt,
#         temperature=0,
#         max_tokens=100,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=["\n"]
#     )["choices"][0]["text"])
#     j += 1
#
# prompt = construct_contextual_prompt(73, [11, 16, 26, 27, 32])
# response = openai.Completion.create(
#     model="text-davinci-002",
#     prompt=prompt,
#     temperature=0,
#     max_tokens=100,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop=["\n"]
# )
# #
# print(prompt + response["choices"][0]["text"])
