from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
import re
import json
from nltk.corpus import gutenberg, stopwords
import nltk
import pickle
# nltk.download('gutenberg')
# nltk.download('stopwords')

filenames = gutenberg.fileids()
stopWords = set(stopwords.words('english'))

with open('../data/scraped_sentences.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('../data/scraped_sentences_validation.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f) 


def text_to_words(inp_text):
    out_text = re.sub('[^A-Za-z]+', ' ', inp_text)  # remove non-alphabets
    # remove single letter words
    out_text = re.sub(r'(?:^| )\w(?:$| )', ' ', out_text).strip()
    out_text = out_text.lower()  # make all characters lower-case
    words = out_text.split()  # tokenise
    filtered_words = []
    for word in words:
        if word not in stopWords:  # discard stop words
            filtered_words.append(word)
    return filtered_words


vocab, tw_map = [], []

for file in filenames:
    word_list = gutenberg.words(file)  # get a list of words from the file
    text = ' '.join(word_list)  # convert into text
    words = text_to_words(text)  # preprocess the text
    vocab += words  # add words to the vocabulary
    tw_map.append(words)  # store the words for each file

topic_sents = list(data1.values())
for sents in topic_sents:  # get a list of sentences for each topic
    text = ' '.join(sents)  # convert into text
    words = text_to_words(text)  # preprocess the text
    vocab += words  # add words to the vocabulary
    tw_map.append(words)  # store the words for each file

topic_sents = list(data2.values())
for sents in topic_sents:  # get a list of sentences for each topic
    text = ' '.join(sents)  # convert into text
    words = text_to_words(text)  # preprocess the text
    vocab += words  # add words to the vocabulary
    tw_map.append(words)  # store the words for each file

count_map = Counter(vocab)

new_vocab = []
for key, val in count_map.items():
    if val > 5:
        new_vocab.append(key)  # keep only words that occur more than 5 times

LE = LabelEncoder()
labels = LE.fit_transform(new_vocab)  # convert words to integer labels
labels = labels.reshape(len(labels), 1)
with open('../word_vectors/vocab_file.pickle', 'wb') as f:
    pickle.dump(dict(zip(LE.classes_, LE.transform(LE.classes_))), f)
VOCAB_SIZE = len(labels)

OE = OneHotEncoder(sparse_output=False)
OE.fit(labels)  # fit one hot encoder onto the words in the vocabulary

vocab_set = set(new_vocab)
new_tw_map = [[word for word in wtype if word in vocab_set]
              for wtype in tw_map]  # remove rare words

word_data = []
for wtype in new_tw_map:
    for i in range(2, len(wtype)-2):
        # store the context window for each word in each text
        window = [wtype[i-2], wtype[i-1], wtype[i+1], wtype[i+2], wtype[i]]
        word_data.append(window)


class CustomDataset(Dataset):
    def __init__(self):
        self.data = np.array(word_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        wlabels = LE.transform(window)  # convert words to integer labels
        wlabels = wlabels.reshape(len(wlabels), 1)
        # convert integer labels to one hot vectors
        wvectors = OE.transform(wlabels)
        wtensors = torch.from_numpy(wvectors)
        # return context words and target word
        return wtensors[:-1], wtensors[-1]

# dataset = CustomDataset()
# data_loader = DataLoader(dataset, batch_size=3, shuffle=True)

# for (idx, batch) in enumerate(data_loader):
# print(batch) # returns context tensor of size (batch_size=3, context_size=4, vocab_size) and target tensor of size (batch_size=3, vocab_size)
# if idx==5: # short test run
# break
