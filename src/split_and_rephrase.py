from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from prepare_pretrain_data import prepare_pretrain_data
import re

rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")
MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]

rfc7252_rule_sentences = []
for sentence in rfc7252:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rfc7252_rule_sentences.append(sentence)
            break
rfc7252_rule_sentences = rfc7252_rule_sentences[1:]

for sentence in rfc7252_rule_sentences:
    if re.search(r"\b" + "when" + r"\b", sentence):
        print(sentence)

model_name = "tuner007/pegasus_paraphrase"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
model.to(device)

batch = tokenizer(["""To avoid an implosion of error responses, when a server is aware that a request arrived via multicast, 
it MUST NOT return a Reset message in reply to a Non-confirmable message."""], truncation=True, padding="longest",
                  return_tensors="pt").to(device)

outputs = tokenizer.batch_decode(
    model.generate(**batch, max_length=100, num_beams=10, num_return_sequences=5, temperature=1.5))
