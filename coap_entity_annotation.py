import string
import nltk
from nltk.corpus import stopwords

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

# # Tokenize the words
# for i in range(len(sentence_text)):
#     sentence_text[i] = nltk.word_tokenize(sentence_text[i], "english")
#
# # Stem the words
# for i in range(len(sentence_text)):
#     for j in range(len(sentence_text[i])):
#         sentence_text[i][j] = nltk.stem.WordNetLemmatizer().lemmatize(sentence_text[i][j])
#
# # Remove stop words and punctuations and record the maximum length
# max_length = 0
# stop_words = set(stopwords.words("english"))
# punctuations = set(string.punctuation)
# punctuations = punctuations.union({"''", "``"})
# stopwords_punctuations = stop_words.union(punctuations)
# for i in range(len(sentence_text)):
#     sentence_text[i] = [word for word in sentence_text[i] if word not in stopwords_punctuations]
#     if len(sentence_text[i]) > max_length:
#         max_length = len(sentence_text[i])
#
# # Pad the sentences to maximum length so that they all have the same length
# PAD = "<PAD>"
# for i in range(len(sentence_text)):
#     if len(sentence_text[i]) < max_length:
#         sentence_text[i].extend((max_length - len(sentence_text[i])) * ["<PAD>"])
#
# # Annotate the words in the sentences with the correct labels (use the index) in a BIO approach
# LABELS = ["B-entity", "I-entity", "O"]
#
#
# def add_trailing_padding(label_list):
#     label_list.extend((max_length - len(label_list)) * [2])
#
#
# y = []
# # 0
# y.append([2, 2, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 0, 1])
# add_trailing_padding(y[0])
# # 1
# y.append([2, 0, 1, 1, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 0, 2, 0])
# add_trailing_padding(y[1])
# # 2
# y.append([0, 1, 0, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 2, 2, 0, 1])
# add_trailing_padding(y[2])
# # 3
# y.append([2, 2, 2, 0, 2, 2, 0])
# add_trailing_padding(y[3])
# # 4
# y.append([2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1])
# add_trailing_padding(y[4])
# # 5
# y.append([2, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 1])
# add_trailing_padding(y[5])
# # 6
# y.append([2, 0, 2, 2, 2, 2, 0, 1, 0, 1, 2, 2, 2, 2, 0, 0, 1, 0, 1, 0, 1, 1])
# add_trailing_padding(y[6])
# # 7
# y.append([2, 2, 0, 1, 1, 0, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1])
# add_trailing_padding(y[7])
# # 8
# y.append([0, 0, 2, 2, 2, 0, 0, 1, 2, 0, 1, 0, 1, 0, 2, 2, 2, 0, 1, 0, 0, 1])
# add_trailing_padding(y[8])
# # 9
# y.append([0, 1, 1])
# add_trailing_padding(y[9])
# # 10
# y.append([0, 1, 1, 0, 1])
# add_trailing_padding(y[10])
# # 11
# y.append([0, 0, 1])
# add_trailing_padding(y[11])
# # 12
# y.append([2, 0, 0, 1])
# add_trailing_padding(y[12])
# # 13
# y.append([2, 0, 1, 2, 0, 2, 2, 2, 0, 1, 2, 0, 2, 2, 0, 2, 2, 2, 2, 0])
# add_trailing_padding(y[13])
# # 14
# y.append([2, 2, 0, 1, 1, 1, 0, 2])
# add_trailing_padding(y[14])
# # 15
# y.append([])
# add_trailing_padding(y[15])
# # 16
# y.append([])
# add_trailing_padding(y[16])
# # 17
# y.append([])
# add_trailing_padding(y[17])
# # 18
# y.append([2, 2, 2, 0])
# add_trailing_padding(y[18])
# # 19
# y.append([2, 2, 2, 2, 2, 0, 2, 2, 0, 1])
# add_trailing_padding(y[19])
# # 20
# y.append([2, 0, 2, 2, 2, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 0, 1])
# add_trailing_padding(y[20])
# # 21
# y.append([0, 2, 0, 0])
# add_trailing_padding(y[21])
# # 22
# y.append([2, 2, 2, 0, 2, 2, 0, 1])
# add_trailing_padding(y[22])
# # 23
# y.append([0, 0, 1, 0])
# add_trailing_padding(y[23])
# # 24
# y.append([2, 2, 2, 0, 2, 2, 0, 1])
# add_trailing_padding(y[24])
# # 25
# y.append([0, 0, 1, 0, 0, 1, 0])
# add_trailing_padding(y[25])
# # 26
# y.append([0, 0, 1, 0, 0, 1, 0])
# add_trailing_padding(y[26])
# # 27
# y.append([0, 1, 0])
# add_trailing_padding(y[27])
# # 28
# y.append([0, 0, 1, 2, 0, 0, 2, 0, 1, 2, 2, 0])
# add_trailing_padding(y[28])
# # 29
# y.append([2, 2, 0, 0, 2, 2, 0])
# add_trailing_padding(y[29])
# # 30
# y.append([0, 0, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2])
# add_trailing_padding(y[30])
# # 31
# y.append([2, 0, 2, 2, 0])
# add_trailing_padding(y[31])
# # 32
# y.append([2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0])
# add_trailing_padding(y[32])
# # 33
# y.append([2, 2, 0, 2, 2, 0, 1, 0, 0, 2, 2, 2, 2, 0])
# add_trailing_padding(y[33])
# # 34
# y.append([0, 0, 2, 0, 2, 2, 2, 0, 2, 2, 0, 2, 0, 2, 2])
# add_trailing_padding(y[34])
# # 35
# y.append([2, 2, 0, 1, 0, 0, 2, 0, 2, 2, 2, 2, 2, 0, 1, 1])
# add_trailing_padding(y[35])
# # 36
# y.append([0, 0, 2, 2, 0, 2, 0])
# add_trailing_padding(y[36])
# # 37
# y.append([2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 1, 0, 1])
# add_trailing_padding(y[37])
# # 38
# y.append([0, 1, 0, 2, 0, 1, 0, 1, 2, 2, 0, 1, 0, 0, 2])
# add_trailing_padding(y[38])
# # 39
# y.append([2, 0])
# add_trailing_padding(y[39])
# # 40
# y.append([0, 0, 1, 0, 2, 0, 2, 2, 0, 0, 1, 1, 0, 1])
# add_trailing_padding(y[40])
# # 41
# y.append([2, 2, 2, 2, 0, 1, 2, 2, 0])
# add_trailing_padding(y[41])
# # 42
# y.append([0, 1, 0, 2, 0])
# add_trailing_padding(y[42])
# # 43
# y.append([0, 2, 0])
# add_trailing_padding(y[43])
# # 44
# y.append([0, 2, 0, 1, 2, 2, 2, 2, 0, 2, 0, 2, 0])
# add_trailing_padding(y[44])
# # 45
# y.append([0, 1, 0, 2, 0])
# add_trailing_padding(y[45])
# # 46
# y.append([2, 2, 0])
# add_trailing_padding(y[46])
# # 47
# y.append([0, 1, 0, 1, 2, 2, 0, 1, 2])
# add_trailing_padding(y[47])
# # 48
# y.append([0, 1, 2, 2, 2, 2, 0, 2, 0, 1, 0, 1, 2, 2, 2, 0, 1, 2])
# add_trailing_padding(y[48])
# # 49
# y.append([0, 1, 0, 1, 2, 2, 0, 0, 0])
# add_trailing_padding(y[49])
# # 50
# y.append([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0])
# add_trailing_padding(y[50])
# # 51
# y.append([2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 2])
# add_trailing_padding(y[51])
# # 52
# y.append([0, 1, 0, 1, 2, 2, 0, 0, 0, 1, 2, 2, 2, 0, 0])
# add_trailing_padding(y[52])
# # 53
# y.append([0, 1, 0, 1, 2, 0, 2, 0, 1, 2, 0, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 1])
# add_trailing_padding(y[53])
# # 54
# y.append([0, 1, 0, 0, 2, 2, 0, 0])
# add_trailing_padding(y[54])
# # 55
# y.append([0, 1, 2, 2, 0])
# add_trailing_padding(y[55])
# # 56
# y.append([0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0])
# add_trailing_padding(y[56])
# # 57
# y.append([2, 0, 0, 1, 2, 0, 2, 2, 2, 2, 0, 1, 2, 0, 1, 0, 1, 0])
# add_trailing_padding(y[57])
# # 58
# y.append([0, 1, 0, 2, 2, 0, 2, 2])
# add_trailing_padding(y[58])
# # 59
# y.append([2, 0, 2, 2, 2, 0])
# add_trailing_padding(y[59])
# # 60
# y.append([0, 1, 0, 2, 2, 2, 0, 2, 0, 2, 2, 2, 0])
# add_trailing_padding(y[60])
# # 61
# y.append([2, 0, 1, 0, 1])
# add_trailing_padding(y[61])
# # 62
# y.append([0, 1, 0, 2, 2, 2, 0])
# add_trailing_padding(y[62])
# # 63
# y.append([2, 0, 2, 2, 2, 0])
# add_trailing_padding(y[63])
# # 64
# y.append([0, 1, 2, 0, 0, 2, 0, 2, 0, 1])
# add_trailing_padding(y[64])
# # 65
# y.append([0, 2, 0, 1, 1, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1])
# add_trailing_padding(y[65])
# # 66
# y.append([2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1])
# add_trailing_padding(y[66])
# # 67
# y.append([2, 2, 0, 1, 0, 1])
# add_trailing_padding(y[67])
# # 68
# add_trailing_padding(y[68])
# # 69
# y.append([2, 2, 0, 2, 0])
# add_trailing_padding(y[69])
# # 70
# y.append([2, 2, 2, 2, 2, 2, 0, 1, 0])
# add_trailing_padding(y[70])
# # 71
# y.append([0, 1, 1, 0, 1, 0, 2, 0, 1, 0])
# add_trailing_padding(y[71])
# # 72
# y.append([2, 0, 1, 2, 2, 0, 1, 2, 0, 0, 2])
# add_trailing_padding(y[72])
# # 73
# y.append([0, 1, 2, 0, 2, 0, 0, 2, 2, 0, 1, 0, 2, 0, 0])
# add_trailing_padding(y[73])
# # 74
# y.append([0, 2, 0, 0, 1, 0, 2, 2, 0, 1])
# add_trailing_padding(y[74])
# # 75
# y.append([2, 0, 0, 2, 2, 2, 0, 1, 0])
# add_trailing_padding(y[75])
# # 76
# y.append([2, 2, 2, 0, 0])
# add_trailing_padding(y[76])
# # 77
# y.append([0, 2, 2, 2, 0, 0, 0, 0, 0])
# add_trailing_padding(y[77])
# # 78
# y.append([0, 1, 0, 1, 2, 0, 2, 2, 0, 0])
# add_trailing_padding(y[78])
# # 79
# y.append([2, 2, 2, 2, 0, 2, 2, 0, 1, 0, 2, 0, 0, 1, 1, 0, 2, 2, 0, 0, 1])
# add_trailing_padding(y[79])
# # 80
# y.append([2, 2, 2, 0, 2, 2, 0, 1, 0, 1, 1, 2, 2, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1])
# add_trailing_padding(y[80])
# # 81
# y.append([0, 2, 2, 0, 0, 0, 2, 0, 1])
# add_trailing_padding(y[81])
# # 82
# y.append([0, 1, 0, 1, 1, 2, 2, 0, 0, 0])
# add_trailing_padding(y[82])
# # 83
# y.append([0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0])
# add_trailing_padding(y[83])
# # 84
# y.append([0, 1, 2, 0, 0])
# add_trailing_padding(y[84])
# # 85
# y.append([0, 1, 1])
# add_trailing_padding(y[85])
# # 86
# y.append([0, 2, 0, 1, 2, 2, 2, 2, 2])
# add_trailing_padding(y[86])
# # 87
# y.append([0, 1, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0, 1])
# add_trailing_padding(y[87])
# # 88
# y.append([0, 2, 2, 0, 0, 0])
# add_trailing_padding(y[88])
# # 89
# y.append([0, 1, 2, 2, 2, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2, 2, 2, 0])
# add_trailing_padding(y[89])
# # 90
# y.append([0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 1, 0, 2, 0, 0])
# add_trailing_padding(y[90])
# # 91
# y.append([2, 2, 0, 1])
# add_trailing_padding(y[91])
# # 92
# y.append([0, 2, 2, 0, 1, 2, 2, 0, 1, 0])
# add_trailing_padding(y[92])
# # 93
# y.append([0, 2, 0, 2, 2, 2, 0, 1, 1, 1, 2, 0, 1, 1])
# add_trailing_padding(y[93])
# # 94
# y.append([2, 2, 2, 2, 2, 0, 1, 0, 1, 2, 2, 0, 2])
# add_trailing_padding(y[94])
# # 95
# y.append([2, 0, 1, 2, 0, 2, 2, 2, 2, 0, 1])
# add_trailing_padding(y[95])
# # 96
# y.append([2, 2, 2, 0, 2, 0, 2, 0, 0, 2, 0])
# add_trailing_padding(y[96])
# # 97
# y.append([0, 1, 0, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 1, 2])
# add_trailing_padding(y[97])
# # 98
# y.append([2, 2, 0, 1, 2, 0, 0, 1, 1, 2, 0, 1])
# add_trailing_padding(y[98])
# # 99
# y.append([0, 2, 2, 0, 0, 2, 2, 0, 2, 2])
# add_trailing_padding(y[99])
# # 100
# y.append([2, 0, 2, 2, 0, 1])
# add_trailing_padding(y[100])
# # 101
# y.append([0, 2, 0, 0, 0, 0, 1, 2, 2, 0, 0, 2, 0, 1, 2, 2, 0, 0, 1])
# add_trailing_padding(y[101])
# # 102
# y.append([2, 0, 1])
# add_trailing_padding(y[102])
# # 103
# y.append([2, 2, 2, 0, 1, 0, 2, 0, 0, 1, 2, 0, 1, 2])
# add_trailing_padding(y[103])
# # 104
# y.append([2, 2, 2, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0])
# add_trailing_padding(y[104])
# # 105
# y.append([0, 2, 0, 2, 2, 0, 1, 2, 2, 2, 0])
# add_trailing_padding(y[105])
# # 106
# y.append([2, 0, 1])
# add_trailing_padding(y[106])
# # 107
# y.append([0, 2, 0, 1, 0, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2, 0, 1])
# add_trailing_padding(y[107])
# # 108
# y.append([])
# add_trailing_padding(y[108])
# # 109
# y.append([2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0])
# add_trailing_padding(y[109])
# # 110
# y.append([0, 2, 2, 2, 2, 0, 2, 2])
# add_trailing_padding(y[110])
# # 111
# y.append([2, 0, 2, 2, 0, 0, 2])
# add_trailing_padding(y[111])
# # 112
# y.append([2, 2, 0, 2, 0, 2, 2, 2, 0, 2, 0, 1, 2, 2, 0, 1, 2])
# add_trailing_padding(y[112])
# # 113
# y.append([0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0])
# add_trailing_padding(y[113])
# # 114
# y.append([0, 1, 2, 2, 2, 0, 1, 1, 0, 1])
# add_trailing_padding(y[114])
# # 115
# y.append([0, 0, 0, 2, 0, 0, 2, 2, 2, 0])
# add_trailing_padding(y[115])
# # 116
# y.append([2, 0, 2, 2, 0, 0, 1, 2, 0, 1])
# add_trailing_padding(y[116])
# # 117
# y.append([0, 2, 2, 0, 0])
# add_trailing_padding(y[117])
# # 118
# y.append([0, 1, 2, 2, 2])
# add_trailing_padding(y[118])
# # 119
# y.append([0, 2, 0, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 2, 0, 2])
# add_trailing_padding(y[119])
# # 120
# y.append([0, 0, 2, 2, 0, 1, 2, 0])
# add_trailing_padding(y[120])
# # 121
