import nltk
import re

nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')


def prepare_pretrain_data(file, author_heading, rfc_heading):
    rfc = open("data/" + file).read()
    rfc = rfc.split("\n")
    rfc_copy = rfc.copy()
    for i in range(len(rfc)):
        line = rfc[i]
        line = line.strip()
        if line.startswith(author_heading) and line.endswith("]"):
            rfc_copy.remove(rfc[i])
        elif line == "":
            rfc_copy.remove(rfc[i])
        elif rfc_heading in line:
            rfc_copy.remove(rfc[i])
    start1 = None
    start2 = None
    end1 = None
    end2 = None
    for i in range(len(rfc_copy)):
        line = rfc_copy[i].strip()
        if line.endswith("Introduction"):
            if not start1:
                start1 = i
                continue
            if start1 and not start2:
                start2 = i
                break

    start = None
    if not start2:
        start = start1
    else:
        start = start2
    rfc_copy = rfc_copy[start:]

    for i in range(len(rfc_copy)):
        line = rfc_copy[i].strip()
        if line.endswith("Acknowledgements") or line.endswith("Acknowledgments"):
            end1 = i
            break
    for i in range(len(rfc_copy)):
        line = rfc_copy[i].strip()
        if line.endswith("References"):
            end2 = i
            break

    for i in range(len(rfc_copy)):
        line = rfc_copy[i].strip()
        if line.startswith("Figure") or line.startswith(("Table")):
            rfc_copy[i] = rfc_copy[i] + "."

    if end1 < end2:
        rfc = "\n".join(rfc_copy[: end1])
    else:
        rfc = "\n".join(rfc_copy[: end2])

    rfc_sentences = nltk.sent_tokenize(rfc, "english")

    for i in range(len(rfc_sentences)):
        rfc_sentences[i] = rfc_sentences[i].replace("\n", "")

        alpha = any(c.isalpha() for c in rfc_sentences[i])
        if not alpha:
            rfc_sentences[i] = ""

        if "Figure" in rfc_sentences[i] and ":" in rfc_sentences[i]:
            rfc_sentences[i] = ""
        if "Table" in rfc_sentences[i] and ":" in rfc_sentences[i]:
            rfc_sentences[i] = ""
        if "+---" in rfc_sentences[i]:
            rfc_sentences[i] = ""
        if "no state!" in rfc_sentences[i]:
            rfc_sentences[i] = ""

    rfc_sentences = [sentence for sentence in rfc_sentences if sentence != ""]

    for i in range(len(rfc_sentences)):
        res = re.sub(' +', ' ', rfc_sentences[i])
        rfc_sentences[i] = res
        if rfc_sentences[i].startswith(" "):
            rfc_sentences[i] = rfc_sentences[i][1:]
        if "- " in rfc_sentences[i]:
            rfc_sentences[i] = rfc_sentences[i].replace("- ", "-")

    return rfc_sentences


rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")

rfc7959 = prepare_pretrain_data("rfc7959.txt", "Bormann & Shelby", "RFC 7959")

rfc8613 = prepare_pretrain_data("rfc8613.txt", "Selander, et al.", "RFC 8613")

rfc8974 = prepare_pretrain_data("rfc8974.txt", "?", "?")

with open(r"./data/pretrain_sentences.txt", "w") as file:
    for sentence in rfc7252:
        file.write("%s\n" % sentence)
    file.write("\n")

    for sentence in rfc7959:
        file.write("%s\n" % sentence)
    file.write("\n")

    for sentence in rfc8613:
        file.write("%s\n" % sentence)
    file.write("\n")

    for sentence in rfc8974:
        file.write("%s\n" % sentence)