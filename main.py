import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pyvi import ViTokenizer
import json


path_to_english_file = 'english.txt'
path_to_vietnamese_file = 'vietnamese.txt'

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    preprocessed_text = ""
    for i in range(len(pos_tags)):
        pos = get_wordnet_pos(pos_tags[i][1])
        lemma = lemmatizer.lemmatize(pos_tags[i][0], pos)
        preprocessed_text += lemma + " "
    return preprocessed_text.strip()

def preprocess_vietnamese_text(text):
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens, tagset='universal')
    preprocessed_text = ""
    for i in range(len(pos_tags)):
        pos = get_wordnet_pos(pos_tags[i][1])
        lemma = lemmatizer.lemmatize(pos_tags[i][0], pos)
        preprocessed_text += lemma + " "
    return preprocessed_text.strip()

english_sentences = []
with open(path_to_english_file, 'r', encoding='utf8') as f:
    for line in f:
        english_sentences.append(preprocess_text(line.strip()))

print("Finished preprocessing English sentences.")

vietnamese_sentences = []
with open(path_to_vietnamese_file, 'r', encoding='utf8') as f:
    for line in f:
        vietnamese_sentences.append(preprocess_vietnamese_text(line.strip()))

print("Finished preprocessing Vietnamese sentences.")


# combine english and vietnamese sentences into a dictionary
data = {'english': english_sentences, 'vietnamese': vietnamese_sentences}

# open a file and write the data to it in JSON format
with open('preprocessed_sentences.json', 'w') as f:
    json.dump(data, f)

print("Finished writing data to JSON file.")
