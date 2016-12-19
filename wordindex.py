import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random

from chatbot.textdata import TextData

with open(os.path.join(os.getcwd() , 'data/samples/dataset-10.pkl'), 'rb') as handle:
    data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    word2id = data["word2id"]
    id2word = data["id2word"]
    trainingSamples = data["trainingSamples"]

    padToken = word2id["<pad>"]
    goToken = word2id["<go>"]
    eosToken = word2id["<eos>"]
    unknownToken = word2id["<unknown>"]  # Restore special words

qmark = word2id['?']
prefixes = list(map(lambda x: word2id[x], ("what", "how", "when", "why", "where", "do", "did", "is", "are", "can", "could", "would", "will")))
filteredSamples = []

for sample in trainingSamples:
    if sample[0][-1] == qmark and sample[0][0] in prefixes:
        filteredSamples.append(sample)

print(len(filteredSamples))
