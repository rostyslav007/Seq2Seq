import spacy
import random
import torch
import torch.nn as nn
import pandas as pd
import torchtext
from nltk.translate.bleu_score import sentence_bleu
from torchtext.vocab import build_vocab_from_iterator
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
from model import Encoder, Decoder, Seq2Seq
import torch.optim as optim

batch_size = 64
num_layers = 2
hidden_size = 1024
embedding_size = 300
dropout_prob = 0.1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# text tokenizers
tok_eng = spacy.load('en_core_web_sm')
tok_ger = spacy.load('de_core_news_md')


def tokenizer_eng(txt):
    return [tok.text for tok in tok_eng.tokenizer(txt)][:-1]


def tokenizer_ger(txt):
    return [tok.text for tok in tok_ger.tokenizer(txt)][:-1]


# data preprocessing
english = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenizer_eng, batch_first=True, eos_token='<eos>', init_token='<sos>')
german = Field(sequential=True, use_vocab=True, lower=True, tokenize=tokenizer_ger, batch_first=True, init_token='<sos>', eos_token='<eos>')
fields = {'English': ('eng', english), 'German': ('ger', german)}
train_data, test_data = TabularDataset.splits(
    path='data/',
    train='train_cleaned.csv',
    test='test_cleaned.csv',
    format='csv',
    fields=fields
)

# building vocabulary
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

# batch training iterator
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=device,
    sort=False
)

input_size_eng = len(english.vocab)
input_size_ger = len(german.vocab)
print(input_size_eng, input_size_ger)

# model components
encoder_attention = Encoder(input_size_eng, 1, hidden_size, embedding_size, dropout_prob, device).to(device=device)
decoder_attention = Decoder(hidden_size, embedding_size, 1, input_size_ger, dropout_prob, device).to(device)
seq2seq_attention = Seq2Seq(encoder_attention, decoder_attention, device).to(device)

# train configs
num_epochs = 100
optimizer = optim.Adam(seq2seq_attention.parameters(), lr=learning_rate)
padding_index = german.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=padding_index)
losses = []

# train loop
for e in range(num_epochs):
    for b in train_iterator:
        e = b.eng.to(device)
        g = b.ger.to(device)

        outputs = seq2seq_attention(e, g).permute(0, 2, 1)
        loss = criterion(outputs, g)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




