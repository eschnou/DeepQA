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

print(id2word[27])
