import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", self.enc_in)

        hidden = getattr(configs, "d_model", 128)   # hidden_size
        n_layers = getattr(configs, "e_layers", 2)  # num_layers
        dropout = getattr(configs, "dropout", 0.0)

        # Esto le da a PROCEED un Linear extra para adaptar (in_proj)
        self.in_proj = nn.Linear(self.enc_in, hidden)

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Linear final (también adaptable)
        self.out_proj = nn.Linear(hidden, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc=None, return_emb=False, *args, **kwargs):
        # x_enc: [B, seq_len, enc_in]
        x = self.in_proj(x_enc)           # [B, seq_len, hidden]
        out, _ = self.lstm(x)             # [B, seq_len, hidden]
        emb = out[:, -1, :]               # [B, hidden]

        y = self.out_proj(emb).view(x_enc.size(0), self.pred_len, self.c_out)
        return (y, emb) if return_emb else y
