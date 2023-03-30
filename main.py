import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from tqdm import tqdm
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import re
from collections import Counter
from vncorenlp import VnCoreNLP
import torchtext
#from torchtext.data import TabularDataset, Field
#from torchtext.legacy import datasets
from sklearn.model_selection import train_test_split
import time
import math

import random
SEED = 2222
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
spacy_en = spacy.load('en_core_web_sm')
annotator = VnCoreNLP("VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

from iteration_utilities import deepflatten
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenize_vi(text):
    return [tok for tok in deepflatten(annotator.tokenize(text), depth=1)]

text_en = 'Please put the dustpan in the broom closet'
text_vi = 'cuốn sách này là của tôi. Của bạn đâu?'
print(tokenize_en(text_en))
print(tokenize_vi(text_vi))

# source = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
# target = Field(tokenize=tokenize_vi, init_token='<sos>', eos_token='<eos>', lower=True)

# fields = {"English": ("src", source), "Vietnamese": ("trg", target)}

# def create_raw_dataset():
#     en_sents = open('english', "r", encoding="utf-8").read().splitlines()
#     vi_sents = open('vietnamese', "r", encoding="utf-8").read().splitlines()

#     return {
#         "English": [line for line in en_sents],
#         "Vietnamese": [line for line in vi_sents],
#     }

# raw_data = create_raw_dataset()

# df = pd.DataFrame(raw_data, columns=["English", "Vietnamese"])
# train, test = train_test_split(df, test_size=0.1)
# train, val = train_test_split(train, test_size=0.125)

# train.to_json("train.json", orient="records", lines=True)
# test.to_json("test.json", orient="records", lines=True)
# val.to_json("val.json", orient="records", lines=True)

# train_data, test_data, val_data = TabularDataset.splits(
#     path="./", train="train.json", test="test.json", validation ="val.json", format="json", fields=fields
# )