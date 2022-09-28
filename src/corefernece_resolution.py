from prepare_pretrain_data import prepare_pretrain_data

rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")

MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL"]
STRONG_MODAL_KEYWORDS = ["MUST", "REQUIRED", "SHALL"]
PRONOUNS = ["It ", " it", "They ", " they ", "Their ", " their", "Them ", " them", "This field",
            "this field", "This value", "this value"]

rule_sentences = []
for sentence in rfc7252:
    for keyword in MODAL_KEYWORDS:
        if keyword in sentence:
            rule_sentences.append(sentence)
            break
rule_sentences = rule_sentences[1:]

pronoun_sentences = []
for sentence in rule_sentences:
    for pronoun in PRONOUNS:
        if pronoun in sentence:
            pronoun_sentences.append(sentence)
            break
