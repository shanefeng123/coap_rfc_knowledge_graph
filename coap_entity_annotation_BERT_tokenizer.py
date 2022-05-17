from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import string
import nltk
import os
from pathlib import Path

# %Don't worry about this part first. This is to train a customized tokenizer. Might be useful later
# rfcs = ["rfc7252.txt", "rfc7959.txt", "rfc8613.txt", "rfc8974.txt"]
# tokenizer = BertWordPieceTokenizer(
#     clean_text=True,
#     handle_chinese_chars=False,
#     strip_accents=False,
#     lowercase=True
# )
#
# tokenizer.train(files=rfcs, vocab_size=30_000, min_frequency=2,
#                 limit_alphabet=1000, wordpieces_prefix='##',
#                 special_tokens=[
#                     '[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
# import os
# from pathlib import Path
# if not os.path.isdir('./bert-coap'):
#   os.mkdir('./bert-coap')
#
# tokenizer.save_model('./bert-coap', 'bert-coap')

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('./bert-coap/bert-coap-vocab.txt')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
rfc7252 = open("rfc7252.txt").read()

# Tokenize sentences
sentence_text = nltk.sent_tokenize(rfc7252, "english")
sentence_text = sentence_text[210:]
# Remove part of the headers
sentence_text = [sentence for sentence in sentence_text if sentence != 'Shelby, et al.']

for i in range(len(sentence_text)):
    # Remove headers of the document
    sentence = sentence_text[i]
    if sentence.startswith("Standards Track"):
        split_position = sentence.find("2014")
        if split_position != -1:
            sentence_text[i] = sentence[split_position + 4:]
        else:
            sentence_text[i] = ""
    sentence_text[i] = sentence_text[i].replace("\n", "")

    # Remove lines contain numbers only
    alpha = any(c.isalpha() for c in sentence_text[i])
    if not alpha:
        sentence_text[i] = ""

    # Remove figures and tables
    if "Figure" in sentence_text[i] and ":" in sentence_text[i]:
        sentence_text[i] = ""
    if "Table" in sentence_text[i] and ":" in sentence_text[i]:
        sentence_text[i] = ""

    # Change to all lower case
    sentence_text[i] = sentence_text[i].lower()

sentence_text = [sentence for sentence in sentence_text if sentence != ""]

# Remove acknowledgement and references
sentence_text = sentence_text[:1282]
sentence_tokens_ids = tokenizer(sentence_text, padding=True)["input_ids"]
sentence_tokens = []
for ids in sentence_tokens_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))
