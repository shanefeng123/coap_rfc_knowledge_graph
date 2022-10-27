from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from prepare_pretrain_data import prepare_pretrain_data
import re

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

conditional_sentences = []
for sentence in rfc7252_rule_sentences:
    for keyword in CONDITIONAL_KEYWORDS:
        if keyword == "if":
            if re.search(r"\b" + keyword.capitalize() + r"\b", sentence) or (
                    re.search(r"\b" + keyword + r"\b", sentence) and
                    not re.search(r"\b" + "even if" + r"\b", sentence)):
                print(sentence)

model_name = "tuner007/pegasus_paraphrase"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
model.to(device)

batch = tokenizer(["""Messages with unknown version numbers MUST be silently ignored.
"""], truncation=True, padding="longest",
                  return_tensors="pt").to(device)

outputs = tokenizer.batch_decode(
    model.generate(**batch, max_length=100, num_beams=10, num_return_sequences=5, temperature=1.5))
