import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0
        self.d_k = d_embed // h
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0)
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        key = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        x = torch.matmul(p_atten, value).transpose(1, 2).contiguous().view(nbatch, -1, self.h * self.d_k)
        return self.linear(x)

# DataLoader creation
def create_data_loader(src_sentences, trg_sentences, src_tokenizer, trg_tokenizer, batch_size):
    dataset = TranslationDataset(src_sentences, trg_sentences, src_tokenizer, trg_tokenizer)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Calculate BLEU score using sacrebleu
def calculate_bleu(predictions, targets, trg_vocab, trg_pad_idx):
    predictions_text = [' '.join([trg_vocab.IdToPiece(int(idx)) for idx in pred if idx != trg_pad_idx]) for pred in predictions]
    targets_text = [[' '.join([trg_vocab.IdToPiece(int(idx)) for idx in target if idx != trg_pad_idx]) for target in targets]]
    bleu = sacrebleu.corpus_bleu(predictions_text, targets_text)
    return bleu.score

# Helper function to calculate the epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
