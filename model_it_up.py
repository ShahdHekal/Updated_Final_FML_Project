import torch
import torch.nn as nn
import torch.nn.functional as F


class LOBDensityModel(nn.Module):
    def __init__(self, tokenizer, embed=32, neurons=256, layers=1):
        super().__init__()
        #embedding so that every feature in the state space is a vector (not a raw number) and we don't 
        #learn arbitrary numerical correlations 
        self.embedding = nn.Embedding(tokenizer.event_size, embed)
        self.encoder = nn.LSTM(embed, neurons, num_layers=layers, batch_first=True)
        self.out = nn.Linear(neurons, tokenizer.event_size)

    def forward(self, tokens):
        x = self.embedding(tokens)
        return self.out(self.encoder(x)[0])

    def loss(self, tokens, reduction="mean"):
        raw_output = self.forward(tokens[:, :-1])
        targets = tokens[:, 1:]
        return F.cross_entropy(raw_output.reshape(-1, raw_output.size(-1)),
                               targets.reshape(-1), reduction=reduction)
