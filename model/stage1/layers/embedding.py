import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(0)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(1, d_model))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # [32, 96, 7]=>[32, 7, 96]
        x = x.unsqueeze(3)  # [32, 7, 96, 1]
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings  # [1, 128]
        return x * y


class InputEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        try:
            x = self.token_embedding(x) + self.pos_embedding(x).to(x.device)
        except:
            import pdb;
            pdb.set_trace()
        return self.dropout(x)