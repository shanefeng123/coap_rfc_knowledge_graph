from transformers import GPT2Tokenizer
import torch
import re
import pdfplumber
import nltk
import pickle

def generate(sentence, model, tokenizer):
    data = f"<|startoftext|>Sentence: {sentence}\nAntecedent:"
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024,
                               pad_token_id=tokenizer.eos_token_id)
    return generated


MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
CONDITIONAL_KEYWORDS = ["if", "when", "unless", "instead", "except", "as", "thus", "therefore", "in case"]
LABELS = ["B-entity", "I-entity", "Other", "PAD"]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model/condition_splitter.pt", map_location=device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")

predictions = []
for i in range(len(mqtt_rule_sentences)):
    rule_sentence = mqtt_rule_sentences[i]
    generated = generate(rule_sentence, model, tokenizer)
    predictions.append(tokenizer.decode(generated[0], skip_special_tokens=True).split("\n", 1)[1])

with open("../data/all_MQTT_condition_split", "wb") as file:
    pickle.dump(predictions, file)