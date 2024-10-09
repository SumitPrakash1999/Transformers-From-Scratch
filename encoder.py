import torch
import torch.nn as nn
from utils import ResidualConnection, MultiHeadedAttention

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadedAttention(config['h'], config['d_embed'], config['dropout'])
        self.feed_forward = nn.Sequential(
            nn.Linear(config['d_embed'], config['d_ff']),
            nn.ReLU(),
            nn.Linear(config['d_ff'], config['d_embed'])
        )
        self.residuals = nn.ModuleList([ResidualConnection(config['d_embed'], config['dropout']) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, mask))
        return self.residuals[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config['encoder_vocab_size'], config['d_embed'])
        self.pos_embed = nn.Parameter(torch.zeros(1, config['max_seq_len'], config['d_embed']))
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config['N_encoder'])])
        self.dropout = nn.Dropout(config['dropout'])
        self.norm = nn.LayerNorm(config['d_embed'])

    def forward(self, x, mask):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
