import torch
import torch.nn as nn
from utils import ResidualConnection, MultiHeadedAttention

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadedAttention(config['h'], config['d_embed'])
        self.attention2 = MultiHeadedAttention(config['h'], config['d_embed'])
        self.feed_forward = nn.Sequential(
            nn.Linear(config['d_embed'], config['d_ff']),
            nn.ReLU(),
            nn.Linear(config['d_ff'], config['d_embed'])
        )
        self.residuals = nn.ModuleList([ResidualConnection(config['d_embed'], config['dropout']) for _ in range(3)])

    def forward(self, memory, src_mask, trg, trg_mask):
        trg = self.residuals[0](trg, lambda x: self.attention1(x, x, x, trg_mask))
        trg = self.residuals[1](trg, lambda x: self.attention2(x, memory, memory, src_mask))
        return self.residuals[2](trg, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(config['decoder_vocab_size'], config['d_embed'])
        self.pos_embed = nn.Parameter(torch.zeros(1, config['max_seq_len'], config['d_embed']))
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config['N_decoder'])])
        self.dropout = nn.Dropout(config['dropout'])
        self.norm = nn.LayerNorm(config['d_embed'])
        self.linear = nn.Linear(config['d_embed'], config['decoder_vocab_size'])

    def forward(self, memory, src_mask, trg, trg_mask):
        trg = self.embedding(trg) + self.pos_embed[:, :trg.size(1), :]
        trg = self.dropout(trg)
        for layer in self.layers:
            trg = layer(memory, src_mask, trg, trg_mask)
        return self.norm(self.linear(trg))
