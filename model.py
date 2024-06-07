import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, d_model: int, dff: int, vocab_size: int,pad_token: int = 0):
        super(Transformer, self).__init__()

        self.pad_token = pad_token
        self.d_model = d_model

        self.embd = nn.Embedding(vocab_size, d_model)

        self.transformer = nn.Transformer(d_model, dim_feedforward=dff, batch_first=True)

        self.ffnn = nn.Linear(d_model, vocab_size)

    def forward(self, enc_in, dec_in):
        enc_pad_mask = self.pad_mask(enc_in, self.pad_token)

        dec_pad_mask = self.look_ahead_masks(dec_in, self.pad_token)

        enc_in = self.embd(enc_in) * math.sqrt(self.d_model)
        dec_in = self.embd(dec_in) * math.sqrt(self.d_model)

        output = self.transformer(enc_in, dec_in, src_key_padding_mask=enc_pad_mask,tgt_mask=dec_pad_mask)

        output = self.ffnn(output)

        return torch.softmax(output, dim=-1)

    def pad_mask(self, seq, pad_token: int = 0):
        mask = (seq == pad_token)
        # mask shape: (batch_size, seq_len)
        return mask.cuda()

    
    def look_ahead_masks(self, seq, pad_token: int = 0):
        batch_size, size = seq.size()
        tgt_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return tgt_mask.cuda()