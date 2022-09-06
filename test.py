# import torch
# import random
from transformers import BertModel, BertTokenizer
bert_uncased = BertModel.from_pretrained("bert-base-uncased")
bert_cased = BertModel.from_pretrained("bert-base-cased")
#
# # bert = BertModel.from_pretrained("bert-base-uncased")
# model = torch.load("coap_BERT.pt")
# device = torch.device("cuda")
# # for parameter in bert.parameters():
# #     print(parameter)
# #     print(parameter.shape)
# #     break
# # for parameter in coap_bert.parameters():
# #     print(parameter)
# #     print(parameter.shape)
# #     break
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# data = open("./data/pretrain_sentences.txt").read().split("\n")
# data_size = len(data)
#
# # Prepare the data for next sentence prediction
# sentences_a = []
# sentences_b = []
# labels = []
#
# # Balance out the data so we have 50% of true next sentences and 50% false next sentences
# for i in range(data_size - 1):
#     sentences_a.append(data[i])
#     if random.random() > 0.5:
#         if not data[i + 1].startswith("introduction"):
#             sentences_b.append(data[i + 1])
#             labels.append(1)
#         else:
#             sentences_b.append(data[random.randint(0, data_size - 1)])
#             labels.append(0)
#     else:
#         sentences_b.append(data[random.randint(0, data_size - 1)])
#         labels.append(0)
#
# # Tokenize the input, create next sentence label tensor, mask 15% of the tokens for mask language modeling
# inputs = tokenizer(sentences_a, sentences_b, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
# inputs["next_sentence_labels"] = torch.LongTensor([labels]).T
#
# inputs["token_labels"] = inputs["input_ids"].detach().clone()
# randomness = torch.rand(inputs["input_ids"].shape)
# mask = (randomness < 0.15) * (inputs["input_ids"] != 101) * (inputs["input_ids"] != 102) * (inputs["input_ids"] != 0)
#
# for i in range(inputs["input_ids"].shape[0]):
#     selections = torch.flatten(mask[i].nonzero()).tolist()
#     inputs["input_ids"][i, selections] = 103
#
# input_ids = inputs["input_ids"].to(device)
# token_type_ids = inputs["token_type_ids"].to(device)
# attention_mask = inputs["attention_mask"].to(device)
# next_sentence_labels = inputs["next_sentence_labels"].to(device)
# token_labels = inputs["token_labels"].to(device)
# outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#                 next_sentence_label=next_sentence_labels, labels=token_labels)
# print(outputs)