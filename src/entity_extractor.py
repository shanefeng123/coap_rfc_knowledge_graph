from transformers import BertTokenizer, AdamW, BertForTokenClassification
import pickle
from tqdm import tqdm
import torch
from prepare_pretrain_data import prepare_pretrain_data


class MeditationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


LABELS = ["B-entity", "I-entity", "Other", "PAD"]

all_entity_indexes = []


def annotate_entity(sentence, entity_indexes):
    annotations = [None] * len(sentence)
    # Annotate all the pad to class 3
    pad_start = sentence.index("[PAD]")
    for i in range(pad_start, len(annotations)):
        annotations[i] = 3

    for entity_index in entity_indexes:
        start = entity_index[0]
        end = entity_index[1]
        annotations[start] = 0
        for i in range(start + 1, end + 1):
            annotations[i] = 1

    for i in range(len(annotations)):
        if annotations[i] is None:
            annotations[i] = 2

    all_entity_indexes.append(entity_indexes)
    return annotations


rfc7252 = prepare_pretrain_data("rfc7252.txt", "Shelby, et al.", "RFC 7252")
rfc7252 = rfc7252[:400]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained("../model/iot_bert", num_labels=len(LABELS))

inputs = tokenizer(rfc7252, padding="max_length", truncation=True, return_tensors="pt")

input_ids = inputs["input_ids"]

input_ids_list = input_ids.tolist()
with open("../data/coap_sentence_input_ids_list", "wb") as file:
    pickle.dump(input_ids_list, file)

sentence_tokens = []
for ids in input_ids:
    sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))

# Annotations
y = []
y.append(annotate_entity(sentence_tokens[0], [(5, 6), (8, 10), (14, 14), (23, 23), (29, 34), (36, 38), (43, 43)]))
y.append(annotate_entity(sentence_tokens[1], [(4, 12), (14, 15), (21, 24), (63, 68)]))
y.append(annotate_entity(sentence_tokens[2], [(1, 4), (7, 12), (18, 20)]))
y.append(annotate_entity(sentence_tokens[3], [(5, 6)]))
y.append(annotate_entity(sentence_tokens[4], [(7, 8), (14, 15), (22, 25), (36, 40), (42, 44)]))
y.append(annotate_entity(sentence_tokens[5], [(4, 5), (13, 15), (29, 31), (34, 36), (42, 45)]))
y.append(annotate_entity(sentence_tokens[6], [(2, 3), (13, 15), (30, 32), (35, 38), (40, 41), (45, 51)]))
y.append(annotate_entity(sentence_tokens[7], [(7, 11), (13, 14), (21, 23), (29, 29), (36, 37), (47, 50), (52, 55)]))
y.append(annotate_entity(sentence_tokens[8], [(2, 3), (11, 12), (14, 16), (19, 22), (24, 25), (38, 40), (42, 43)]))
y.append(annotate_entity(sentence_tokens[9], [(2, 8)]))
y.append(annotate_entity(sentence_tokens[10], []))
y.append(annotate_entity(sentence_tokens[11], [(2, 3), (5, 7)]))
y.append(annotate_entity(sentence_tokens[12], [(3, 4), (6, 7)]))
y.append(annotate_entity(sentence_tokens[13], [(5, 7), (11, 13), (20, 21), (22, 22), (24, 26), (33, 35), (43, 44)]))
y.append(annotate_entity(sentence_tokens[14], [(5, 10), (12, 14)]))
y.append(annotate_entity(sentence_tokens[15], []))
y.append(annotate_entity(sentence_tokens[16], []))
y.append(annotate_entity(sentence_tokens[17], [(26, 26), (34, 34)]))
y.append(annotate_entity(sentence_tokens[18], [(10, 12)]))
y.append(annotate_entity(sentence_tokens[19], [(11, 12), (18, 19)]))
y.append(annotate_entity(sentence_tokens[20], [(7, 8), (19, 19), (35, 39), (45, 48), (51, 52)]))
y.append(annotate_entity(sentence_tokens[21], [(1, 2), (4, 6), (9, 9)]))
y.append(annotate_entity(sentence_tokens[22], [(9, 10), (17, 19)]))
y.append(annotate_entity(sentence_tokens[23], [(1, 4), (6, 8), (11, 11)]))
y.append(annotate_entity(sentence_tokens[24], [(9, 9), (16, 18)]))
y.append(annotate_entity(sentence_tokens[25], [(1, 2), (4, 6), (9, 9), (12, 14), (17, 17)]))
y.append(annotate_entity(sentence_tokens[26], [(1, 1), (3, 5), (8, 8), (11, 13), (16, 16)]))
y.append(annotate_entity(sentence_tokens[27], [(1, 2), (4, 4), (9, 9)]))
y.append(annotate_entity(sentence_tokens[28], [(1, 3), (5, 8), (14, 14), (18, 18), (21, 22), (27, 29)]))
y.append(annotate_entity(sentence_tokens[29], [(6, 8), (11, 12), (18, 20)]))
y.append(annotate_entity(sentence_tokens[30], [(1, 2), (4, 6), (14, 14), (19, 19), (23, 24), (26, 28), (31, 32)]))
y.append(annotate_entity(sentence_tokens[31], [(4, 6), (12, 14)]))
y.append(annotate_entity(sentence_tokens[32], [(11, 11), (21, 22), (24, 27), (29, 32)]))
y.append(annotate_entity(sentence_tokens[33], [(7, 8), (13, 14), (16, 19), (22, 25), (35, 35)]))
y.append(annotate_entity(sentence_tokens[34], [(1, 4), (6, 7), (11, 11), (21, 21), (26, 26)]))
y.append(annotate_entity(sentence_tokens[35], [(9, 10), (11, 11), (14, 15), (17, 19), (23, 23), (32, 35)]))
y.append(annotate_entity(sentence_tokens[36], [(1, 5), (7, 8), (17, 17), (25, 25)]))
y.append(annotate_entity(sentence_tokens[37], [(3, 6), (9, 9), (20, 23), (26, 29), (31, 31), (37, 38), (41, 42)]))
y.append(annotate_entity(sentence_tokens[38], [(1, 9), (11, 12), (17, 19), (22, 24), (33, 35), (39, 39), (42, 42)]))
y.append(annotate_entity(sentence_tokens[39], [(5, 8)]))
y.append(annotate_entity(sentence_tokens[40], [(1, 4), (6, 10), (14, 17), (24, 25), (35, 44), (47, 56)]))
y.append(annotate_entity(sentence_tokens[41], [(9, 18), (26, 30)]))
y.append(annotate_entity(sentence_tokens[42], [(1, 5), (7, 7), (10, 11)]))
y.append(annotate_entity(sentence_tokens[43], [(2, 2), (6, 9)]))
y.append(annotate_entity(sentence_tokens[44], [(3, 3), (8, 12), (18, 19), (22, 27), (30, 31)]))
y.append(annotate_entity(sentence_tokens[45], [(1, 5), (8, 8), (13, 14)]))
y.append(annotate_entity(sentence_tokens[46], [(6, 6)]))
y.append(annotate_entity(sentence_tokens[47], [(1, 7), (9, 15), (21, 25)]))
y.append(annotate_entity(sentence_tokens[48], [(5, 11), (20, 20), (27, 31), (35, 41), (46, 50)]))
y.append(annotate_entity(sentence_tokens[49], [(1, 3), (5, 7), (12, 12), (14, 17), (19, 22)]))
y.append(annotate_entity(sentence_tokens[50], [(8, 9), (26, 26)]))
y.append(annotate_entity(sentence_tokens[51], [(4, 6), (16, 21), (36, 37), (40, 43)]))
y.append(annotate_entity(sentence_tokens[52], [(1, 5), (7, 11), (17, 24), (26, 27), (29, 29), (39, 40), (43, 43)]))
y.append(annotate_entity(sentence_tokens[53], [(1, 3), (6, 10), (13, 13), (18, 19), (28, 28), (40, 42), (47, 49)]))
y.append(annotate_entity(sentence_tokens[54], [(1, 2), (4, 4), (7, 7), (9, 11), (15, 15), (18, 18)]))
y.append(annotate_entity(sentence_tokens[55], [(2, 3), (11, 11)]))
y.append(annotate_entity(sentence_tokens[56], [(1, 3), (5, 5), (14, 15), (19, 19), (26, 26)]))
y.append(annotate_entity(sentence_tokens[57], [(6, 7), (14, 15), (26, 27), (31, 32), (34, 35), (38, 38)]))
y.append(annotate_entity(sentence_tokens[58], [(1, 5), (7, 7), (16, 17)]))
y.append(annotate_entity(sentence_tokens[59], [(4, 4), (9, 9)]))
y.append(annotate_entity(sentence_tokens[60], [(1, 4), (6, 6), (15, 16), (19, 19), (26, 26)]))
y.append(annotate_entity(sentence_tokens[61], [(3, 4), (7, 9)]))
y.append(annotate_entity(sentence_tokens[62], [(1, 7), (9, 9), (21, 22)]))
y.append(annotate_entity(sentence_tokens[63], [(4, 4), (9, 9)]))
y.append(annotate_entity(sentence_tokens[64], [(1, 2), (7, 9), (13, 13), (18, 19)]))
y.append(annotate_entity(sentence_tokens[65], [(1, 4), (9, 11), (21, 23), (29, 32), (38, 43), (48, 53)]))
y.append(annotate_entity(sentence_tokens[66], [(8, 13), (31, 32)]))
y.append(annotate_entity(sentence_tokens[67], []))
y.append(annotate_entity(sentence_tokens[68], []))
y.append(annotate_entity(sentence_tokens[69], []))
y.append(annotate_entity(sentence_tokens[70], []))
y.append(annotate_entity(sentence_tokens[71], [(1, 5), (7, 8), (10, 11), (16, 19), (21, 23)]))
y.append(annotate_entity(sentence_tokens[72], [(3, 8), (13, 14), (19, 19), (21, 21)]))
y.append(annotate_entity(sentence_tokens[73], [(2, 4), (10, 12), (18, 18), (26, 27), (31, 31), (36, 37), (41, 41)]))
y.append(annotate_entity(sentence_tokens[74], [(2, 2), (6, 6), (9, 10), (13, 13)]))
y.append(annotate_entity(sentence_tokens[75], [(2, 4), (6, 7), (21, 25), (28, 29)]))
y.append(annotate_entity(sentence_tokens[76], [(10, 10), (13, 14), (17, 23)]))
y.append(annotate_entity(sentence_tokens[77], [(1, 2), (7, 7), (9, 12), (14, 17), (19, 24), (26, 27)]))
y.append(annotate_entity(sentence_tokens[78], [(1, 3), (5, 7), (13, 13), (17, 17), (19, 19)]))
y.append(annotate_entity(sentence_tokens[79], [(9, 9), (18, 21), (23, 23), (28, 31), (33, 37), (40, 40), (54, 60)]))
y.append(annotate_entity(sentence_tokens[80], [(5, 6), (18, 22), (27, 28), (43, 46), (48, 48), (50, 52)]))
y.append(annotate_entity(sentence_tokens[81], [(1, 2), (10, 11), (13, 15), (21, 23)]))
y.append(annotate_entity(sentence_tokens[82], [(1, 4), (6, 10), (17, 17), (19, 20), (22, 23)]))
y.append(annotate_entity(sentence_tokens[83], [(1, 2), (10, 10), (23, 23), (26, 26)]))
y.append(annotate_entity(sentence_tokens[84], [(2, 3), (7, 7), (8, 8)]))
y.append(annotate_entity(sentence_tokens[85], [(2, 5)]))
y.append(annotate_entity(sentence_tokens[86], [(2, 2), (5, 6), (15, 16)]))
y.append(annotate_entity(sentence_tokens[87], [(3, 4), (18, 18), (23, 24), (29, 30)]))
y.append(annotate_entity(sentence_tokens[88], [(1, 3), (9, 9), (11, 14), (16, 17)]))
y.append(annotate_entity(sentence_tokens[89],
                         [(2, 6), (16, 17), (19, 25), (27, 31), (35, 35), (38, 44), (46, 47), (52, 53), (68, 69)]))
y.append(annotate_entity(sentence_tokens[90], [(3, 3), (12, 16), (30, 31), (38, 40), (42, 43), (48, 53), (55, 56)]))
y.append(annotate_entity(sentence_tokens[91], [(2, 2), (7, 8), (29, 33), (35, 36)]))
y.append(annotate_entity(sentence_tokens[92], [(10, 11)]))
y.append(annotate_entity(sentence_tokens[93], [(3, 3), (10, 14), (21, 23), (25, 26)]))
y.append(annotate_entity(sentence_tokens[94], [(7, 9)]))
y.append(annotate_entity(sentence_tokens[95], [(2, 3), (6, 7), (15, 19), (22, 26)]))
y.append(annotate_entity(sentence_tokens[96], [(8, 10), (12, 14), (22, 23)]))
y.append(annotate_entity(sentence_tokens[97], [(2, 3), (7, 8), (17, 20)]))
y.append(annotate_entity(sentence_tokens[98], [(9, 11), (20, 22), (24, 25), (30, 35)]))
y.append(annotate_entity(sentence_tokens[99], [(1, 5), (6, 8), (10, 10), (16, 18), (24, 25), (27, 28)]))
y.append(annotate_entity(sentence_tokens[100], [(7, 7), (9, 9), (15, 16), (18, 20), (24, 26)]))
y.append(annotate_entity(sentence_tokens[101], [(2, 3), (8, 8), (10, 10), (15, 15)]))
y.append(annotate_entity(sentence_tokens[102], [(5, 6), (13, 14)]))
y.append(annotate_entity(sentence_tokens[103],
                         [(2, 2), (7, 10), (12, 13), (16, 19), (21, 22), (24, 24), (33, 33), (36, 36), (40, 44),
                          (50, 55), (57, 58), (60, 60)]))
y.append(annotate_entity(sentence_tokens[104], [(5, 9)]))
y.append(annotate_entity(sentence_tokens[105], [(10, 14), (18, 18), (25, 25), (28, 34), (37, 41)]))
y.append(annotate_entity(sentence_tokens[106], [(6, 8), (10, 14), (28, 30), (32, 33), (35, 35)]))
y.append(annotate_entity(sentence_tokens[107], [(3, 3), (12, 12), (16, 20), (27, 34), (38, 38), (47, 47)]))
y.append(annotate_entity(sentence_tokens[108], [(3, 3), (8, 8), (14, 18), (30, 30)]))
y.append(annotate_entity(sentence_tokens[109], [(6, 7)]))
y.append(annotate_entity(sentence_tokens[110], [(3, 3), (8, 12), (16, 16), (22, 26), (30, 30), (35, 39)]))
y.append(annotate_entity(sentence_tokens[111], []))
y.append(
    annotate_entity(sentence_tokens[112], [(1, 2), (3, 4), (8, 9), (11, 12), (14, 16), (19, 21), (22, 22), (28, 30)]))
y.append(annotate_entity(sentence_tokens[113], [(9, 11), (30, 33), (40, 42)]))
y.append(annotate_entity(sentence_tokens[114], [(1, 2), (11, 12)]))
y.append(annotate_entity(sentence_tokens[115], [(2, 2), (9, 9), (11, 11)]))
y.append(annotate_entity(sentence_tokens[116], [(4, 4), (8, 8), (12, 12), (21, 23), (31, 34), (36, 39)]))
y.append(
    annotate_entity(sentence_tokens[117], [(1, 2), (6, 6), (11, 11), (16, 17), (22, 22), (24, 24), (26, 26), (29, 30)]))
y.append(annotate_entity(sentence_tokens[118], [(1, 3), (10, 14), (18, 22)]))
y.append(annotate_entity(sentence_tokens[119], [(1, 3), (5, 7), (12, 13), (15, 15), (21, 21)]))
y.append(annotate_entity(sentence_tokens[120], [(2, 3), (7, 8), (10, 11), (14, 16)]))
y.append(annotate_entity(sentence_tokens[121], [(2, 2), (8, 9), (12, 14)]))
y.append(annotate_entity(sentence_tokens[122], [(1, 4)]))
y.append(annotate_entity(sentence_tokens[123], [(1, 3), (7, 10), (18, 19), (27, 27)]))
y.append(annotate_entity(sentence_tokens[124], [(2, 4), (6, 6), (11, 14)]))
y.append(annotate_entity(sentence_tokens[125], [(4, 5), (8, 9), (12, 12), (14, 14), (19, 19), (23, 25), (33, 34)]))
y.append(annotate_entity(sentence_tokens[126], [(10, 12)]))
y.append(annotate_entity(sentence_tokens[127],
                         [(2, 3), (9, 12), (14, 16), (28, 31), (40, 41), (43, 45), (48, 50), (52, 53)]))
y.append(annotate_entity(sentence_tokens[128], [(10, 16), (18, 19), (24, 26), (28, 29)]))
y.append(annotate_entity(sentence_tokens[129], [(9, 13), (16, 19), (26, 26), (28, 29), (31, 32), (35, 35), (39, 42)]))
y.append(annotate_entity(sentence_tokens[130], [(7, 10)]))
y.append(annotate_entity(sentence_tokens[131], [(1, 2), (3, 4), (8, 13), (19, 23)]))
y.append(annotate_entity(sentence_tokens[132], [(1, 3), (4, 5), (13, 13), (22, 23), (31, 33), (36, 37), (40, 43)]))
y.append(annotate_entity(sentence_tokens[133], [(1, 2), (8, 13), (15, 17)]))
y.append(annotate_entity(sentence_tokens[134], [(8, 8), (11, 11), (13, 14), (17, 18)]))
y.append(annotate_entity(sentence_tokens[135], [(2, 6), (13, 17), (28, 29)]))
y.append(annotate_entity(sentence_tokens[136], [(1, 3), (9, 10)]))
y.append(annotate_entity(sentence_tokens[137], [(2, 3), (14, 14)]))
y.append(annotate_entity(sentence_tokens[138], [(9, 11)]))
y.append(annotate_entity(sentence_tokens[139], [(3, 5), (13, 16), (18, 22), (24, 26), (35, 35), (43, 44)]))
y.append(annotate_entity(sentence_tokens[140], [(5, 5), (11, 11), (13, 14)]))
y.append(annotate_entity(sentence_tokens[141], [(5, 8)]))
y.append(annotate_entity(sentence_tokens[142], []))
y.append(annotate_entity(sentence_tokens[143], []))
y.append(annotate_entity(sentence_tokens[144], [(1, 2), (5, 6)]))
y.append(annotate_entity(sentence_tokens[145], [(1, 1), (3, 3)]))
y.append(annotate_entity(sentence_tokens[146], [(6, 6), (10, 13), (18, 21), (26, 31), (37, 38)]))
y.append(annotate_entity(sentence_tokens[147], [(6, 7)]))
y.append(annotate_entity(sentence_tokens[148], [(1, 3), (5, 7)]))
y.append(annotate_entity(sentence_tokens[149], [(11, 12)]))
y.append(annotate_entity(sentence_tokens[150], [(25, 27)]))
y.append(annotate_entity(sentence_tokens[151], [(1, 1), (16, 16), (27, 27)]))
y.append(annotate_entity(sentence_tokens[152], [(2, 2), (6, 6), (12, 13), (19, 21), (28, 30)]))
y.append(annotate_entity(sentence_tokens[153], [(4, 5)]))
y.append(annotate_entity(sentence_tokens[154], [(6, 9), (12, 13)]))
y.append(annotate_entity(sentence_tokens[155], [(5, 5), (8, 8), (12, 14), (20, 20), (23, 24)]))
y.append(annotate_entity(sentence_tokens[156], [(9, 13)]))
y.append(annotate_entity(sentence_tokens[157], [(5, 5), (7, 7)]))
y.append(annotate_entity(sentence_tokens[158], [(1, 2), (11, 14)]))
y.append(annotate_entity(sentence_tokens[159], [(4, 6), (10, 10), (13, 18), (20, 21), (23, 23), (26, 29), (31, 34)]))
y.append(annotate_entity(sentence_tokens[160], [(6, 7), (10, 10)]))
y.append(annotate_entity(sentence_tokens[161], [(2, 2), (7, 9), (24, 26)]))
y.append(annotate_entity(sentence_tokens[162], [(2, 4), (11, 11), (13, 13)]))
y.append(annotate_entity(sentence_tokens[163], [(6, 7), (12, 12), (14, 14)]))
y.append(annotate_entity(sentence_tokens[164], [(1, 2), (4, 5), (12, 13)]))
y.append(annotate_entity(sentence_tokens[165], [(2, 3), (12, 12), (16, 17), (22, 25), (28, 28)]))
y.append(annotate_entity(sentence_tokens[166], [(3, 3), (5, 5), (8, 8), (16, 16)]))
y.append(annotate_entity(sentence_tokens[167], [(22, 25), (37, 37), (43, 43)]))
y.append(annotate_entity(sentence_tokens[168], [(2, 2), (8, 8), (14, 17), (25, 27), (32, 33)]))
y.append(annotate_entity(sentence_tokens[169], [(5, 8), (14, 14)]))
y.append(annotate_entity(sentence_tokens[170], [(5, 5), (12, 12), (20, 22)]))
y.append(annotate_entity(sentence_tokens[171], [(18, 19), (21, 21), (42, 43)]))
y.append(annotate_entity(sentence_tokens[172], [(12, 13), (20, 20)]))
y.append(annotate_entity(sentence_tokens[173], [(1, 4), (5, 6), (11, 11), (18, 18)]))
y.append(annotate_entity(sentence_tokens[174], [(2, 2), (6, 6), (11, 13), (17, 19), (25, 27), (31, 33)]))
y.append(annotate_entity(sentence_tokens[175], [(6, 8), (21, 23), (26, 26), (34, 36), (47, 47), (50, 52), (59, 59)]))
y.append(annotate_entity(sentence_tokens[176], [(7, 7), (11, 11), (14, 16)]))
y.append(annotate_entity(sentence_tokens[177], [(6, 6), (13, 13)]))
y.append(annotate_entity(sentence_tokens[178], [(1, 3), (9, 13)]))
y.append(annotate_entity(sentence_tokens[179], [(12, 12)]))
y.append(annotate_entity(sentence_tokens[180], [(5, 5), (11, 13)]))
y.append(annotate_entity(sentence_tokens[181], [(9, 11)]))
y.append(annotate_entity(sentence_tokens[182], [(27, 29)]))
y.append(annotate_entity(sentence_tokens[183], [(11, 14), (23, 25)]))
y.append(annotate_entity(sentence_tokens[184], [(7, 10)]))
y.append(annotate_entity(sentence_tokens[185], [(17, 18), (28, 30)]))
y.append(annotate_entity(sentence_tokens[186], [(3, 5), (13, 15), (18, 18), (24, 24), (31, 31)]))
y.append(annotate_entity(sentence_tokens[187], [(6, 8), (16, 18), (25, 25)]))
y.append(annotate_entity(sentence_tokens[188], [(1, 3)]))
y.append(annotate_entity(sentence_tokens[189], [(12, 14)]))
y.append(annotate_entity(sentence_tokens[190], [(23, 25), (29, 31)]))
y.append(annotate_entity(sentence_tokens[191], [(11, 14), (19, 21), (25, 27)]))
y.append(annotate_entity(sentence_tokens[192], []))
y.append(annotate_entity(sentence_tokens[193], [(18, 20)]))
y.append(annotate_entity(sentence_tokens[194], [(1, 1), (7, 9)]))
y.append(annotate_entity(sentence_tokens[195], [(7, 9), (14, 14)]))
y.append(annotate_entity(sentence_tokens[196], [(14, 14), (25, 27)]))
y.append(annotate_entity(sentence_tokens[197], [(1, 5), (7, 7), (17, 19)]))
y.append(annotate_entity(sentence_tokens[198], [(1, 1)]))
y.append(annotate_entity(sentence_tokens[199], [(1, 2), (5, 6)]))
y.append(annotate_entity(sentence_tokens[200], [(1, 2), (13, 16), (26, 28)]))
y.append(annotate_entity(sentence_tokens[201], [(2, 2), (23, 24)]))
y.append(annotate_entity(sentence_tokens[202], [(11, 13)]))
y.append(annotate_entity(sentence_tokens[203], [(2, 2)]))
y.append(annotate_entity(sentence_tokens[204], [(12, 13)]))
y.append(annotate_entity(sentence_tokens[205], [(1, 1), (4, 6), (12, 15), (22, 26)]))
y.append(annotate_entity(sentence_tokens[206], [(11, 15), (20, 22), (44, 48)]))
y.append(annotate_entity(sentence_tokens[207], [(14, 16), (20, 23), (38, 40)]))
y.append(annotate_entity(sentence_tokens[208], [(4, 7), (22, 31)]))
y.append(annotate_entity(sentence_tokens[209], [(1, 3), (4, 6), (16, 19)]))
y.append(annotate_entity(sentence_tokens[210], [(6, 8), (10, 10)]))
y.append(annotate_entity(sentence_tokens[211], [(2, 3), (11, 12), (14, 16)]))
y.append(annotate_entity(sentence_tokens[212], [(5, 6), (10, 11), (27, 28)]))
y.append(annotate_entity(sentence_tokens[213], [(9, 18), (20, 26), (28, 32)]))
y.append(annotate_entity(sentence_tokens[214], [(2, 4), (7, 10), (12, 16)]))
y.append(annotate_entity(sentence_tokens[215], [(1, 2), (4, 5), (7, 10), (13, 13), (15, 15), (18, 20)]))
y.append(annotate_entity(sentence_tokens[216], [(6, 7), (11, 11), (15, 16)]))
y.append(annotate_entity(sentence_tokens[217], [(3, 3), (10, 11), (17, 18), (27, 28), (31, 32), (38, 39), (42, 45)]))
y.append(annotate_entity(sentence_tokens[218], [(3, 4), (7, 8), (15, 16)]))
y.append(annotate_entity(sentence_tokens[219], [(6, 6)]))
y.append(annotate_entity(sentence_tokens[220], [(5, 5), (10, 10), (14, 17)]))
y.append(annotate_entity(sentence_tokens[221], [(5, 6), (9, 9), (13, 13), (16, 16), (20, 20)]))
y.append(annotate_entity(sentence_tokens[222], [(6, 10), (14, 17), (23, 26)]))
y.append(annotate_entity(sentence_tokens[223], [(12, 16)]))
y.append(annotate_entity(sentence_tokens[224], [(2, 3), (6, 6), (10, 12)]))
y.append(annotate_entity(sentence_tokens[225], [(2, 4), (27, 28)]))
y.append(annotate_entity(sentence_tokens[226], [(16, 18)]))
y.append(annotate_entity(sentence_tokens[227], [(1, 2), (11, 11), (14, 14), (20, 20), (22, 25), (28, 30)]))
y.append(annotate_entity(sentence_tokens[228], [(2, 6), (11, 11), (13, 13), (24, 26), (33, 33)]))
y.append(annotate_entity(sentence_tokens[229],
                         [(2, 2), (12, 16), (19, 25), (32, 32), (35, 35), (41, 41), (48, 48), (50, 50), (54, 54),
                          (71, 73)]))
y.append(annotate_entity(sentence_tokens[230], [(5, 9), (17, 19)]))
y.append(annotate_entity(sentence_tokens[231], [(2, 8), (14, 15), (18, 22), (29, 29), (32, 32)]))
y.append(annotate_entity(sentence_tokens[232], [(2, 4), (10, 11), (14, 18), (24, 24)]))
y.append(annotate_entity(sentence_tokens[233], [(5, 10), (12, 14), (21, 26), (29, 29), (32, 32), (40, 42), (45, 45)]))
y.append(annotate_entity(sentence_tokens[234], [(4, 4), (6, 11), (13, 15), (24, 29), (31, 33)]))
y.append(annotate_entity(sentence_tokens[235], [(2, 3), (10, 14), (27, 28), (31, 33)]))
y.append(annotate_entity(sentence_tokens[236], [(1, 4), (12, 15), (24, 28), (35, 36), (39, 40), (44, 45), (48, 52)]))
y.append(annotate_entity(sentence_tokens[237], [(4, 8), (12, 13), (30, 37), (40, 58), (70, 74)]))
y.append(annotate_entity(sentence_tokens[238], [(3, 4), (9, 13), (17, 25), (28, 28), (37, 41), (49, 50)]))
y.append(annotate_entity(sentence_tokens[239], [(3, 7), (9, 17), (20, 21), (26, 27), (30, 32), (40, 40)]))
y.append(annotate_entity(sentence_tokens[240], [(8, 9), (12, 13), (17, 17)]))
y.append(annotate_entity(sentence_tokens[241], [(19, 25)]))
y.append(annotate_entity(sentence_tokens[242], [(5, 6)]))
y.append(annotate_entity(sentence_tokens[243], [(9, 12), (14, 21), (32, 32), (41, 51), (67, 68)]))
y.append(annotate_entity(sentence_tokens[244], [(2, 5), (9, 13), (23, 24), (28, 36)]))
y.append(annotate_entity(sentence_tokens[245], [(9, 9), (16, 16), (26, 28)]))
y.append(annotate_entity(sentence_tokens[246], [(5, 8), (15, 16), (26, 27), (31, 32), (37, 40), (43, 43)]))
y.append(annotate_entity(sentence_tokens[247], [(4, 5), (22, 22), (40, 41), (44, 44), (52, 56), (62, 63)]))
y.append(annotate_entity(sentence_tokens[248], [(6, 9), (18, 20)]))
y.append(annotate_entity(sentence_tokens[249],
                         [(9, 12), (18, 22), (39, 40), (43, 46), (49, 50), (52, 54), (58, 59), (61, 61), (63, 64),
                          (67, 68), (80, 83), (85, 88)]))
y.append(annotate_entity(sentence_tokens[250], [(1, 5), (22, 24), (65, 69)]))
y.append(annotate_entity(sentence_tokens[251], [(1, 3), (5, 9), (10, 13)]))
y.append(annotate_entity(sentence_tokens[252], [(1, 1), (3, 3), (5, 5), (8, 13), (15, 17)]))
y.append(annotate_entity(sentence_tokens[253], [(1, 2), (11, 11), (16, 17)]))
y.append(annotate_entity(sentence_tokens[254], [(6, 6)]))
y.append(annotate_entity(sentence_tokens[255], [(8, 8), (19, 19), (21, 24)]))
y.append(annotate_entity(sentence_tokens[256], [(2, 6), (11, 11), (13, 13), (21, 21)]))
y.append(annotate_entity(sentence_tokens[257], [(2, 6), (16, 16)]))
y.append(annotate_entity(sentence_tokens[258], [(2, 2), (8, 8), (16, 16), (24, 24), (26, 26), (30, 30), (47, 49)]))
y.append(annotate_entity(sentence_tokens[259], [(5, 9), (16, 18), (24, 26), (29, 29)]))
y.append(annotate_entity(sentence_tokens[260], [(3, 4), (13, 14), (19, 23)]))
y.append(annotate_entity(sentence_tokens[261], [(2, 3), (13, 17), (19, 29), (45, 51), (54, 54), (66, 66)]))
y.append(annotate_entity(sentence_tokens[262], [(4, 4), (11, 11), (13, 17), (20, 21)]))
y.append(annotate_entity(sentence_tokens[263], [(3, 4), (14, 16), (18, 22)]))
y.append(annotate_entity(sentence_tokens[264], [(17, 18)]))
y.append(annotate_entity(sentence_tokens[265], [(20, 22), (25, 28)]))
y.append(annotate_entity(sentence_tokens[266], [(1, 1), (6, 11), (13, 15), (20, 24), (26, 30), (35, 36), (45, 46)]))
y.append(annotate_entity(sentence_tokens[267], [(2, 3), (17, 18), (21, 24), (26, 30), (35, 37)]))
y.append(annotate_entity(sentence_tokens[268], [(2, 3), (11, 16), (18, 20), (23, 23)]))
y.append(annotate_entity(sentence_tokens[269], [(3, 4), (19, 20), (24, 34)]))
y.append(annotate_entity(sentence_tokens[270], [(14, 16)]))
y.append(annotate_entity(sentence_tokens[271], [(7, 10), (12, 14), (19, 20), (30, 33), (35, 39)]))
y.append(annotate_entity(sentence_tokens[272], [(1, 2), (12, 13), (22, 23)]))
y.append(annotate_entity(sentence_tokens[273], [(5, 7), (14, 16), (18, 19), (26, 27), (29, 31)]))
y.append(annotate_entity(sentence_tokens[274], []))
y.append(annotate_entity(sentence_tokens[275],
                         [(3, 8), (10, 12), (16, 19), (21, 25), (28, 29), (31, 33), (36, 41), (43, 45), (51, 52),
                          (54, 56), (59, 62), (64, 68)]))
y.append(annotate_entity(sentence_tokens[276],
                         [(1, 4), (6, 6), (11, 15), (21, 22), (24, 26), (32, 42), (57, 62), (71, 73), (77, 78)]))
y.append(annotate_entity(sentence_tokens[277], [(2, 2), (14, 18), (22, 27), (29, 31), (39, 39), (41, 41), (44, 44)]))
y.append(annotate_entity(sentence_tokens[278], [(10, 14), (17, 17)]))
y.append(annotate_entity(sentence_tokens[279], [(4, 7), (11, 11), (19, 23), (30, 30), (34, 34), (52, 54)]))
y.append(annotate_entity(sentence_tokens[280], [(10, 12), (15, 16), (18, 19), (22, 24), (25, 25), (28, 28), (45, 45)]))
y.append(annotate_entity(sentence_tokens[281], [(6, 6), (22, 22)]))
y.append(annotate_entity(sentence_tokens[282], [(9, 12), (25, 25), (42, 42), (54, 54)]))
y.append(annotate_entity(sentence_tokens[283], [(2, 2), (7, 11), (17, 18), (20, 22), (27, 34)]))
y.append(annotate_entity(sentence_tokens[284], [(18, 18), (21, 21), (32, 36), (44, 44), (46, 46), (49, 49)]))
y.append(annotate_entity(sentence_tokens[285], [(1, 3), (13, 15), (22, 25)]))
y.append(annotate_entity(sentence_tokens[286], [(2, 3), (13, 14)]))
y.append(annotate_entity(sentence_tokens[287], [(1, 2), (6, 7), (13, 15)]))
y.append(annotate_entity(sentence_tokens[288], [(2, 4), (20, 21), (29, 31), (39, 41), (50, 52)]))
y.append(annotate_entity(sentence_tokens[289], [(3, 5), (11, 11), (14, 16), (38, 39), (51, 52), (60, 61)]))
y.append(annotate_entity(sentence_tokens[290], [(6, 7), (12, 13), (18, 20), (28, 30)]))
y.append(annotate_entity(sentence_tokens[291], [(5, 7), (19, 21)]))
y.append(annotate_entity(sentence_tokens[292], [(2, 4), (23, 25), (26, 28), (51, 53), (55, 57), (74, 75), (78, 80)]))
y.append(annotate_entity(sentence_tokens[293], [(15, 20), (26, 29), (46, 47)]))
y.append(annotate_entity(sentence_tokens[294],
                         [(6, 7), (10, 13), (18, 19), (26, 34), (61, 62), (67, 70), (72, 75), (82, 83)]))
y.append(annotate_entity(sentence_tokens[295], [(1, 2)]))
y.append(annotate_entity(sentence_tokens[296], [(11, 11), (14, 14)]))
y.append(annotate_entity(sentence_tokens[297], [(30, 30), (33, 36), (44, 45), (48, 48), (64, 65)]))
y.append(annotate_entity(sentence_tokens[298], [(6, 8), (10, 10), (17, 17), (25, 25)]))
y.append(annotate_entity(sentence_tokens[299], [(2, 2), (8, 8), (12, 14), (16, 22), (34, 35), (38, 38)]))
y.append(annotate_entity(sentence_tokens[300], [(2, 2), (9, 9), (13, 13), (20, 20), (24, 24), (31, 33), (35, 38)]))
y.append(annotate_entity(sentence_tokens[301], [(1, 4), (6, 7), (9, 10), (15, 21)]))
y.append(annotate_entity(sentence_tokens[302], [(8, 8), (11, 13), (24, 25), (32, 32), (35, 37), (40, 43)]))
y.append(
    annotate_entity(sentence_tokens[303], [(2, 3), (7, 8), (12, 13), (24, 25), (29, 29), (34, 34), (37, 43), (65, 66)]))
y.append(annotate_entity(sentence_tokens[304], [(5, 8)]))
y.append(annotate_entity(sentence_tokens[305], [(2, 5), (23, 26), (41, 44)]))
y.append(annotate_entity(sentence_tokens[306], [(2, 12), (15, 15), (19, 19), (22, 26), (30, 36)]))
y.append(annotate_entity(sentence_tokens[307], [(7, 7), (14, 14), (17, 21), (29, 33)]))
y.append(annotate_entity(sentence_tokens[308], [(7, 10), (24, 25), (30, 32), (34, 40), (45, 46)]))
y.append(annotate_entity(sentence_tokens[309], [(3, 4), (10, 11), (15, 15)]))
y.append(annotate_entity(sentence_tokens[310], [(3, 3), (12, 12), (21, 24)]))
y.append(annotate_entity(sentence_tokens[311], [(18, 18), (25, 26), (29, 30)]))
y.append(annotate_entity(sentence_tokens[312], [(7, 8), (19, 20)]))
y.append(annotate_entity(sentence_tokens[313], [(9, 16), (18, 27), (29, 37), (39, 42), (44, 52), (61, 67)]))
y.append(annotate_entity(sentence_tokens[314], []))
y.append(annotate_entity(sentence_tokens[315], [(2, 3), (21, 21)]))
y.append(annotate_entity(sentence_tokens[316], [(3, 3), (16, 16), (20, 22)]))
y.append(annotate_entity(sentence_tokens[317], [(7, 14)]))
y.append(annotate_entity(sentence_tokens[318], []))
y.append(annotate_entity(sentence_tokens[319], [(1, 2), (13, 17), (19, 20)]))
y.append(annotate_entity(sentence_tokens[320], [(10, 17), (21, 24)]))
y.append(annotate_entity(sentence_tokens[321], [(1, 4), (11, 18), (21, 24), (30, 32), (38, 38)]))
y.append(annotate_entity(sentence_tokens[322], [(1, 10)]))
y.append(annotate_entity(sentence_tokens[323], [(1, 9), (28, 32)]))
y.append(annotate_entity(sentence_tokens[324], [(5, 6), (13, 14), (26, 27), (41, 42)]))
y.append(annotate_entity(sentence_tokens[325], [(1, 3), (8, 12), (16, 23), (25, 34), (37, 45), (50, 54)]))
y.append(annotate_entity(sentence_tokens[326], [(13, 14), (24, 34), (45, 49), (53, 56)]))
y.append(annotate_entity(sentence_tokens[327],
                         [(4, 5), (31, 38), (45, 53), (59, 68), (70, 80), (91, 95), (101, 102), (108, 109),
                          (111, 112)]))
y.append(annotate_entity(sentence_tokens[328], [(4, 5), (33, 40), (48, 56), (65, 74), (89, 89)]))
y.append(annotate_entity(sentence_tokens[329], [(2, 8), (14, 15)]))
y.append(annotate_entity(sentence_tokens[330], [(7, 8), (10, 13)]))
y.append(annotate_entity(sentence_tokens[331], [(9, 19), (22, 28)]))
y.append(annotate_entity(sentence_tokens[332], [(11, 17)]))
y.append(annotate_entity(sentence_tokens[333], [(10, 10), (20, 21), (27, 28)]))
y.append(annotate_entity(sentence_tokens[334], []))
y.append(annotate_entity(sentence_tokens[335], []))
y.append(annotate_entity(sentence_tokens[336], [(2, 10), (21, 25), (28, 29)]))
y.append(annotate_entity(sentence_tokens[337], [(10, 11), (15, 16), (30, 37)]))
y.append(annotate_entity(sentence_tokens[338], [(2, 6), (9, 13), (20, 26), (29, 37), (55, 65), (74, 78), (84, 85)]))
y.append(annotate_entity(sentence_tokens[339], [(1, 11), (14, 24), (27, 33), (36, 44), (48, 54)]))
y.append(annotate_entity(sentence_tokens[340],
                         [(9, 19), (22, 22), (32, 39), (45, 53), (59, 69), (71, 81), (86, 92), (102, 108)]))
y.append(annotate_entity(sentence_tokens[341], [(5, 15), (22, 32), (37, 43), (46, 54), (61, 62)]))
y.append(annotate_entity(sentence_tokens[342], [(2, 9), (16, 20), (25, 26)]))
y.append(annotate_entity(sentence_tokens[343], [(6, 8), (16, 22)]))
y.append(annotate_entity(sentence_tokens[344], [(4, 7), (11, 13), (20, 22)]))
y.append(annotate_entity(sentence_tokens[345], [(24, 24), (32, 42)]))
y.append(annotate_entity(sentence_tokens[346], [(15, 25), (27, 33), (40, 41), (59, 61), (69, 79)]))
y.append(annotate_entity(sentence_tokens[347], []))
y.append(annotate_entity(sentence_tokens[348], []))
y.append(annotate_entity(sentence_tokens[349],
                         [(1, 4), (8, 9), (14, 17), (19, 21), (24, 27), (34, 34), (40, 42), (46, 46), (52, 52),
                          (55, 57)]))
y.append(annotate_entity(sentence_tokens[350], [(2, 4), (6, 8), (27, 29)]))
y.append(
    annotate_entity(sentence_tokens[351], [(1, 3), (5, 7), (11, 11), (17, 17), (20, 22), (28, 28), (30, 32), (44, 44)]))
y.append(annotate_entity(sentence_tokens[352], [(1, 2), (6, 6), (8, 9), (11, 13), (15, 16), (19, 21), (28, 30)]))
y.append(annotate_entity(sentence_tokens[353], [(33, 35)]))
y.append(annotate_entity(sentence_tokens[354], [(2, 4)]))
y.append(annotate_entity(sentence_tokens[355], [(2, 3), (5, 6), (9, 11)]))
y.append(annotate_entity(sentence_tokens[356], [(1, 3), (18, 19), (24, 25), (38, 39)]))
y.append(annotate_entity(sentence_tokens[357], [(2, 2), (8, 9), (12, 14), (17, 20), (23, 27), (30, 31), (34, 34)]))
y.append(annotate_entity(sentence_tokens[358], [(2, 2), (5, 5)]))
y.append(annotate_entity(sentence_tokens[359],
                         [(1, 2), (9, 9), (12, 12), (16, 18), (24, 24), (29, 32), (47, 48), (52, 56), (59, 64)]))
y.append(annotate_entity(sentence_tokens[360], [(2, 2), (7, 8), (11, 13), (18, 19)]))
y.append(annotate_entity(sentence_tokens[361], [(4, 8), (11, 14), (26, 26)]))
y.append(annotate_entity(sentence_tokens[362], []))
y.append(annotate_entity(sentence_tokens[363], [(2, 3), (10, 11), (14, 16), (21, 26)]))
y.append(annotate_entity(sentence_tokens[364], [(10, 11), (15, 17)]))
y.append(annotate_entity(sentence_tokens[365], []))
y.append(annotate_entity(sentence_tokens[366], [(15, 17), (21, 22)]))
y.append(annotate_entity(sentence_tokens[367], [(5, 6), (11, 13)]))
y.append(annotate_entity(sentence_tokens[368], [(6, 7), (10, 10), (12, 12), (15, 15)]))
y.append(annotate_entity(sentence_tokens[369], [(1, 1), (3, 7), (10, 10)]))
y.append(annotate_entity(sentence_tokens[370], [(1, 1), (3, 6), (9, 9), (16, 16)]))
y.append(annotate_entity(sentence_tokens[371],
                         [(2, 4), (13, 15), (18, 22), (24, 27), (38, 39), (48, 49), (54, 56), (58, 60)]))
y.append(annotate_entity(sentence_tokens[372], [(7, 8), (14, 15), (18, 18), (29, 30), (39, 39)]))
y.append(annotate_entity(sentence_tokens[373], [(3, 5)]))
y.append(annotate_entity(sentence_tokens[374], [(1, 2)]))
y.append(annotate_entity(sentence_tokens[375], [(1, 4), (12, 12), (18, 24), (29, 29), (35, 35), (40, 44)]))
y.append(annotate_entity(sentence_tokens[376], [(6, 10)]))
y.append(annotate_entity(sentence_tokens[377], [(2, 2), (7, 13), (19, 19)]))
y.append(annotate_entity(sentence_tokens[378], [(5, 5), (7, 10), (13, 19), (23, 24), (30, 30)]))
y.append(annotate_entity(sentence_tokens[379], [(13, 15), (17, 17), (29, 29), (33, 33)]))
y.append(annotate_entity(sentence_tokens[380], [(2, 2)]))
y.append(annotate_entity(sentence_tokens[381], [(16, 16), (21, 23), (37, 37), (41, 41)]))
y.append(annotate_entity(sentence_tokens[382], [(1, 2), (11, 15)]))
y.append(annotate_entity(sentence_tokens[383], [(5, 5), (25, 31), (37, 37), (45, 46)]))
y.append(annotate_entity(sentence_tokens[384], [(2, 2), (5, 5), (9, 13), (23, 29)]))
y.append(annotate_entity(sentence_tokens[385], [(8, 8)]))
y.append(annotate_entity(sentence_tokens[386], [(2, 2), (8, 9), (20, 24)]))
y.append(annotate_entity(sentence_tokens[387], [(6, 7), (14, 14)]))
y.append(annotate_entity(sentence_tokens[388], [(3, 3), (14, 14)]))
y.append(annotate_entity(sentence_tokens[389],
                         [(7, 7), (17, 21), (24, 24), (27, 27), (32, 32), (35, 40), (45, 46), (50, 50)]))
y.append(annotate_entity(sentence_tokens[390], [(9, 13)]))
y.append(annotate_entity(sentence_tokens[391], [(3, 3), (8, 9), (14, 19), (22, 26), (29, 30)]))
y.append(annotate_entity(sentence_tokens[392], [(3, 3), (7, 13), (24, 24), (27, 32), (37, 37), (45, 45)]))
y.append(annotate_entity(sentence_tokens[393], [(8, 8), (16, 21), (27, 33), (39, 39), (47, 48)]))
y.append(annotate_entity(sentence_tokens[394],
                         [(3, 3), (7, 11), (14, 14), (17, 22), (25, 25), (32, 33), (40, 40), (43, 43)]))
y.append(annotate_entity(sentence_tokens[395], [(2, 2), (13, 13), (17, 22), (27, 28), (30, 30), (33, 35)]))
y.append(annotate_entity(sentence_tokens[396], [(12, 14), (23, 27), (30, 30), (38, 44), (47, 47), (66, 67)]))
y.append(annotate_entity(sentence_tokens[397], [(7, 9), (25, 25)]))
y.append(annotate_entity(sentence_tokens[398], [(6, 7), (21, 22), (35, 44), (48, 48), (57, 57)]))
y.append(annotate_entity(sentence_tokens[399], [(1, 4), (7, 8), (10, 13), (17, 17), (26, 30)]))

with open("../data/coap_sentence_entity_labels", "wb") as file:
    pickle.dump(y, file)

with open("../data/coap_sentence_entity_indexes", "wb") as file:
    pickle.dump(all_entity_indexes, file)

labels = torch.LongTensor(y)
inputs["labels"] = labels


def train(batch, model, optimizer):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                          labels=labels)
    train_loss = train_outputs.loss
    train_loss.backward()
    optimizer.step()
    predictions = torch.argmax(train_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / (
            labels.shape[0] * labels.shape[1])
    return train_loss, predictions, accuracy, labels


def test(batch, model):
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                         labels=labels)
    test_loss = test_outputs.loss
    predictions = torch.argmax(test_outputs.logits, dim=-1)
    accuracy = torch.sum(torch.eq(predictions, labels)) / (
            labels.shape[0] * labels.shape[1])
    return test_loss, predictions, accuracy, labels


dataset = MeditationDataset(inputs)
dataset_length = len(dataset)
train_length = int(dataset_length * 0.8)
test_length = dataset_length - train_length
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_length, test_length])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cpu":
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(20):
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    epoch_train_predictions = None
    epoch_train_labels = None
    num_of_train_batches = len(train_loader)
    for train_batch in train_loop:
        model.train()
        train_loss, train_predictions, train_accuracy, train_labels = train(train_batch, model, optimizer)
        train_loop.set_postfix(train_loss=train_loss.item(), train_accuracy=train_accuracy.item())
        overall_train_loss += train_loss.item()
        train_loop.set_description(f"Epoch {epoch} train")

        if epoch_train_predictions is None:
            epoch_train_predictions = train_predictions
            epoch_train_labels = train_labels
        else:
            epoch_train_predictions = torch.cat((epoch_train_predictions, train_predictions), dim=0)
            epoch_train_labels = torch.cat((epoch_train_labels, train_labels), dim=0)

    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    epoch_test_predictions = None
    epoch_test_labels = None
    num_of_test_batches = len(test_loader)
    for test_batch in test_loop:
        model.eval()
        test_loss, test_predictions, test_accuracy, test_labels = test(test_batch, model)
        test_loop.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())
        overall_test_loss += test_loss.item()
        test_loop.set_description(f"Epoch {epoch} test")

        if epoch_test_predictions is None:
            epoch_test_predictions = test_predictions
            epoch_test_labels = test_labels
        else:
            epoch_test_predictions = torch.cat((epoch_test_predictions, test_predictions), dim=0)
            epoch_test_labels = torch.cat((epoch_test_labels, test_labels), dim=0)

    average_train_loss = overall_train_loss / num_of_train_batches
    epoch_train_accuracy = torch.sum(torch.eq(epoch_train_predictions, epoch_train_labels)) / (
            epoch_train_labels.shape[0] * epoch_train_labels.shape[1])

    average_test_loss = overall_test_loss / num_of_test_batches
    epoch_test_accuracy = torch.sum(torch.eq(epoch_test_predictions, epoch_test_labels)) / (
            epoch_test_labels.shape[0] * epoch_test_labels.shape[1])

    print(f"average train loss: {average_train_loss}")
    print(f"epoch train accuracy: {epoch_train_accuracy.item()}")

    print(f"average test loss: {average_test_loss}")
    print(f"epoch test accuracy: {epoch_test_accuracy.item()}")

    with open(r"../results/entity_extractor_results.txt", "a") as file:
        file.write(
            f"Epoch {epoch} average_train_loss: {average_train_loss} train_accuracy: {epoch_train_accuracy.item()}")
        file.write("\n")
        file.write(
            f"Epoch {epoch} average_test_loss: {average_test_loss} test_accuracy: {epoch_test_accuracy.item()}")
        file.write("\n")

    if epoch_train_accuracy.item() > 0.99 and epoch_test_accuracy.item() > 0.99:
        break

torch.save(model, r"../model/entity_extractor.pt")
# test_sentence = "MQTT is an OASIS standard messaging protocol for the Internet of Things (IoT)."
# test_inputs = tokenizer(test_sentence, return_tensors="pt")
# test_input_ids = test_inputs["input_ids"].to(device)
# test_token_type_ids = test_inputs["token_type_ids"].to(device)
# test_attention_mask = test_inputs["attention_mask"].to(device)
# test_outputs = model(input_ids=test_input_ids, token_type_ids=test_token_type_ids, attention_mask=test_attention_mask)
# test_predictions = torch.argmax(test_outputs.logits, dim=-1)
# test_sentence_tokens = []
# for ids in test_input_ids:
#     test_sentence_tokens.append(tokenizer.convert_ids_to_tokens(ids))
# print(test_sentence_tokens)
# print(test_predictions)
