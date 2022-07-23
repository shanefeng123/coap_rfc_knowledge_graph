import nltk
import re

nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')

# Handling rfc7252
rfc7252 = open("data/rfc7252.txt").read()

# Use nltk to tokenize sentences
rfc7252_sentence_text = nltk.sent_tokenize(rfc7252, "english")
rfc7252_sentence_text = rfc7252_sentence_text[210:]
# Remove part of the headers
rfc7252_sentence_text = [sentence for sentence in rfc7252_sentence_text if sentence != 'Shelby, et al.']

for i in range(len(rfc7252_sentence_text)):
    # Remove headers of the document
    sentence = rfc7252_sentence_text[i]
    if sentence.startswith("Standards Track"):
        split_position = sentence.find("2014")
        if split_position != -1:
            rfc7252_sentence_text[i] = sentence[split_position + 4:]
        else:
            rfc7252_sentence_text[i] = ""
    rfc7252_sentence_text[i] = rfc7252_sentence_text[i].replace("\n", "")

    # Remove lines contain numbers only
    alpha = any(c.isalpha() for c in rfc7252_sentence_text[i])
    if not alpha:
        rfc7252_sentence_text[i] = ""

    # Remove figures and tables
    if "Figure" in rfc7252_sentence_text[i] and ":" in rfc7252_sentence_text[i]:
        rfc7252_sentence_text[i] = ""
    if "Table" in rfc7252_sentence_text[i] and ":" in rfc7252_sentence_text[i]:
        rfc7252_sentence_text[i] = ""

    # Change to all lower case
    rfc7252_sentence_text[i] = rfc7252_sentence_text[i].lower()

    if "+---" in rfc7252_sentence_text[i]:
        rfc7252_sentence_text[i] = ""

for i in range(len(rfc7252_sentence_text)):
    if rfc7252_sentence_text[i].endswith("shelby, et al."):
        rfc7252_sentence_text[i] = rfc7252_sentence_text[i][0:rfc7252_sentence_text[i].find("shelby, et al.")] + rfc7252_sentence_text[i + 1]
        rfc7252_sentence_text[i + 1] = ""

rfc7252_sentence_text = [sentence for sentence in rfc7252_sentence_text if sentence != ""]

# Remove acknowledgement and references
rfc7252_sentence_text = rfc7252_sentence_text[:1269]

for i in range(len(rfc7252_sentence_text)):
    res = re.sub(' +', ' ', rfc7252_sentence_text[i])
    rfc7252_sentence_text[i] = res
    if rfc7252_sentence_text[i].startswith(" "):
        rfc7252_sentence_text[i] = rfc7252_sentence_text[i][1:]

with open(r"./data/pretrain_sentences.txt", "w") as file:
    for sentence in rfc7252_sentence_text:
        file.write("%s\n" % sentence)
    file.write("\n")



# Read in RFC7959
rfc7959 = open("./data/rfc7959.txt").read()
rfc7959_sentence_text = nltk.sent_tokenize(rfc7959, "english")
# Start from introduction
rfc7959_sentence_text = rfc7959_sentence_text[72:]
# Remove headers
for i in range(len(rfc7959_sentence_text)):
    sentence = rfc7959_sentence_text[i]
    if sentence.startswith("Bormann & Shelby"):
        split_position = sentence.find("2016") + 4
        rfc7959_sentence_text[i] = sentence[split_position:]

    # Remove lines contain numbers only
    alpha = any(c.isalpha() for c in rfc7959_sentence_text[i])
    if not alpha:
        rfc7959_sentence_text[i] = ""

    # Remove tables and figures
    if "Figure" in rfc7959_sentence_text[i] and ":" in rfc7959_sentence_text[i]:
        rfc7959_sentence_text[i] = ""
    if "Table" in rfc7959_sentence_text[i] and ":" in rfc7959_sentence_text[i]:
        rfc7959_sentence_text[i] = ""

    # Remove tables or figures left over
    if "+---" in rfc7959_sentence_text[i]:
        rfc7959_sentence_text[i] = ""

    # Change to all lower case
    rfc7959_sentence_text[i] = rfc7959_sentence_text[i].lower()

    # Get rid of new line characters
    rfc7959_sentence_text[i] = rfc7959_sentence_text[i].replace("\n", "")

rfc7959_sentence_text = [sentence for sentence in rfc7959_sentence_text if sentence != ""]
rfc7959_sentence_text = rfc7959_sentence_text[:267]

for i in range(len(rfc7959_sentence_text)):
    res = re.sub(' +', ' ', rfc7959_sentence_text[i])
    rfc7959_sentence_text[i] = res
    if rfc7959_sentence_text[i].startswith(" "):
        rfc7959_sentence_text[i] = rfc7959_sentence_text[i][1:]

with open(r"./data/pretrain_sentences.txt", "a") as file:
    for sentence in rfc7959_sentence_text:
        file.write("%s\n" % sentence)
    file.write("\n")

# Read in RFC8613
rfc8613 = open("./data/rfc8613.txt").read()
rfc8613_sentence_text = nltk.sent_tokenize(rfc8613, "english")
# Start from introduction
rfc8613_sentence_text = rfc8613_sentence_text[146:]
for i in range(len(rfc8613_sentence_text)):
    if rfc8613_sentence_text[i].startswith("Selander"):
        rfc8613_sentence_text[i] = ""
    if rfc8613_sentence_text[i].startswith("Standards Track"):
        split_position = rfc8613_sentence_text[i].find("2019") + 4
        rfc8613_sentence_text[i] = rfc8613_sentence_text[i][split_position:]

    rfc8613_sentence_text[i] = rfc8613_sentence_text[i].replace("\n", "")

    # Remove lines contain numbers only
    alpha = any(c.isalpha() for c in rfc8613_sentence_text[i])
    if not alpha:
        rfc8613_sentence_text[i] = ""

    # Remove tables and figures
    if "Figure" in rfc8613_sentence_text[i] and ":" in rfc8613_sentence_text[i]:
        rfc8613_sentence_text[i] = ""
    if "Table" in rfc8613_sentence_text[i] and ":" in rfc8613_sentence_text[i]:
        rfc8613_sentence_text[i] = ""

    # Remove tables or figures left over
    if "+---" in rfc8613_sentence_text[i]:
        rfc8613_sentence_text[i] = ""

    # Change to all lower case
    rfc8613_sentence_text[i] = rfc8613_sentence_text[i].lower()

rfc8613_sentence_text = [sentence for sentence in rfc8613_sentence_text if sentence != ""]
rfc8613_sentence_text = rfc8613_sentence_text[:671]

for i in range(len(rfc8613_sentence_text)):
    res = re.sub(' +', ' ', rfc8613_sentence_text[i])
    rfc8613_sentence_text[i] = res
    if rfc8613_sentence_text[i].startswith(" "):
        rfc8613_sentence_text[i] = rfc8613_sentence_text[i][1:]

with open(r"./data/pretrain_sentences.txt", "a") as file:
    for sentence in rfc8613_sentence_text:
        file.write("%s\n" % sentence)
    file.write("\n")

# Read in RFC8974
rfc8974 = open("./data/rfc8974.txt").read()
rfc8974_sentence_text = nltk.sent_tokenize(rfc8974, "english")
# Start from introduction
rfc8974_sentence_text = rfc8974_sentence_text[44:]

for i in range(len(rfc8974_sentence_text)):
    rfc8974_sentence_text[i] = rfc8974_sentence_text[i].replace("\n", "")
    # Remove lines contain numbers only
    alpha = any(c.isalpha() for c in rfc8974_sentence_text[i])
    if not alpha:
        rfc8974_sentence_text[i] = ""

    # Remove tables and figures
    if "Figure" in rfc8974_sentence_text[i] and ":" in rfc8974_sentence_text[i]:
        rfc8974_sentence_text[i] = ""
    if "Table" in rfc8974_sentence_text[i] and ":" in rfc8974_sentence_text[i]:
        rfc8974_sentence_text[i] = ""

    # Remove tables or figures left over
    if "|   " in rfc8974_sentence_text[i]:
        rfc8974_sentence_text[i] = ""

    # Change to all lower case
    rfc8974_sentence_text[i] = rfc8974_sentence_text[i].lower()

rfc8974_sentence_text = [sentence for sentence in rfc8974_sentence_text if sentence != ""]
rfc8974_sentence_text = rfc8974_sentence_text[:155]

for i in range(len(rfc8974_sentence_text)):
    res = re.sub(' +', ' ', rfc8974_sentence_text[i])
    rfc8974_sentence_text[i] = res
    if rfc8974_sentence_text[i].startswith(" "):
        rfc8974_sentence_text[i] = rfc8974_sentence_text[i][1:]

with open(r"./data/pretrain_sentences.txt", "a") as file:
    for sentence in rfc8974_sentence_text:
        file.write("%s\n" % sentence)