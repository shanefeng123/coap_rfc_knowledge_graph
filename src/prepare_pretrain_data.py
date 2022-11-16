import nltk
import pdfplumber
import re

nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')


def prepare_pretrain_data(file, author_heading, rfc_heading):
    """
    This is a generic function to extract the relevant content of an RFC document
    Args:
        file: RFC file name
        author_heading: The author heading that needs to be excluded
        rfc_heading: The RFC heading that needs to be excluded

    Returns: List of sentences from the RFC document

    """
    rfc = open("../data/" + file).read()
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

# MQTT specification is a PDF document, so it needs to be parsed specifically
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
# This is just to ignore some references in the specification
mqtt_spec_sentences = mqtt_spec_sentences[:46] + mqtt_spec_sentences[49:]


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

# dss_spec = []
# with pdfplumber.open("../data/dss_specification.pdf") as pdf:
#     pages = pdf.pages[12:]
#     for page in pages:
#         text = page.extract_text(layout=False)
#         print(text)
#         break


# Write these sentences to a file, with a new line character to separate different documents
with open(r"../data/pretrain_sentences.txt", "w") as file:
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
    file.write("\n")

    for sentence in mqtt_spec_sentences:
        file.write("%s\n" % sentence)
    file.write("\n")

    for sentence in amqp_spec_sentences:
        file.write("%s\n" % sentence)


