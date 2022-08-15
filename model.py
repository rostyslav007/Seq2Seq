import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, embedding_size, dropour_prob, device):
        super(Encoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropour_prob)
        self.embedding_layer = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True,
                           bidirectional=True)
        self.fc_hidden = nn.Linear(2*hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_hidden = nn.Linear(2*hidden_size, hidden_size)
        self.fc_cell = nn.Linear(2*hidden_size, hidden_size)

        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for p in m.parameters():
                    p.data.uniform_(-0.08, 0.08)

    def forward(self, x):
        x = self.dropout(self.embedding_layer(x))
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, device=self.device)
        encoder_states, (hidden, cell) = self.rnn(x, (h0, c0))
        hidden = self.fc_hidden(torch.cat([hidden[0], hidden[1]], dim=-1))
        cell = self.fc_cell(torch.cat([cell[0], cell[1]], dim=-1))
        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers, vocab_size, dropout_p, device):
        super(Decoder, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(2*hidden_size + embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.energy_layer = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
                for p in m.parameters():
                    p.data.uniform_(-0.08, 0.08)

    def forward(self, encoder_states, x, hidden, cell):
        seq_len = encoder_states.size(1)
        batch_s = x.size(0)
        # x of shape (N, 1) as we pass 1 word at tume to our Decoder
        x = self.dropout(self.embedding_layer(x))
        hidden_reshaped = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        mutual_state = torch.cat([encoder_states, hidden_reshaped], dim=-1).reshape((-1, self.hidden_size*3))
        energy = self.relu(self.energy_layer(mutual_state).reshape((batch_s, seq_len)))
        attention = self.softmax(energy).unsqueeze(-1)
        weighted_encoder_states = encoder_states * attention
        c_i = torch.sum(weighted_encoder_states, dim=1).unsqueeze(1)
        x_concat = torch.cat([c_i, x], dim=-1)
        output, (hidden_next, cell_next) = self.rnn(x_concat, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        predictions = self.fc(output)
        hidden_next = hidden_next.squeeze(0)
        cell_next = cell_next.squeeze(0)
        return predictions, hidden_next, cell_next


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, target_word_ratio=0.5):
        seq_len = source.size(1)
        targ_len = target.size(1)
        encoder_states, hidden, cell = self.encoder(source)
        outputs = torch.zeros(source.size(0), targ_len, self.decoder.vocab_size,
                              device=self.device)
        x = source[:, 0].unsqueeze(1)

        for i in range(1, targ_len):
            x, hidden, cell = self.decoder(encoder_states, x, hidden, cell)
            outputs[:, i] = x.squeeze(1)
            preds = x.argmax(dim=-1).to(device=self.device)
            x = target[:, i].unsqueeze(1) if random.random() < target_word_ratio else preds

        return outputs
