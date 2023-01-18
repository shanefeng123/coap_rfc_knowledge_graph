from transformers import GPT2Tokenizer
import torch
import re
from tqdm import tqdm
import pickle
from nltk.translate.bleu_score import sentence_bleu
import nltk
import pdfplumber

def generate(context, sentence, model, tokenizer):
    data = f"<|startoftext|>Context: {context}\nSentence: {sentence}\nProperties: "
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :512]
        attention_mask = attention_mask[:, :512]

    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024,
                           pad_token_id=tokenizer.eos_token_id)
    return generated



def construct_context(pronoun_sentence, specification_sentences, k):
    pronoun_sentence_index = specification_sentences.index(pronoun_sentence)
    context_start_index = pronoun_sentence_index - k
    context_sentences = specification_sentences[context_start_index:pronoun_sentence_index + 1]
    return " ".join(context_sentences)


MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
LABELS = ["B-entity", "I-entity", "Other", "PAD"]

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

rule_condition_split = pickle.load(open('../data/all_AMQP_condition_split', 'rb'))

contexts = []
# Construct context for the split sentences
for i in range(len(rule_condition_split)):
    sentence_split = rule_condition_split[i].split("\n")
    original_sentence = amqp_rule_sentences[i]
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model/properties_extractor.pt")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")

all_properties = []
for i in range(len(contexts)):
    context = contexts[i][0]
    sentence = contexts[i][1]
    generated = generate(context, sentence, model, tokenizer)
    all_properties.append(tokenizer.decode(generated[0], skip_special_tokens=True).split("Properties:")[1].strip())

with open("../data/amqp_contexts_condition_split", "wb") as file:
    pickle.dump(contexts, file)

with open("../data/all_AMQP_properties", "wb") as file:
    pickle.dump(all_properties, file)