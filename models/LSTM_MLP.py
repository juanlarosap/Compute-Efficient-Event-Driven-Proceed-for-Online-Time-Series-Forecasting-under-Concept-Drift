import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LSTM + MLP(2 capas) compatible con Online + PROCEED.
    forward(x_enc, x_mark_enc=None, return_emb=False) -> y o (y, emb)
    y:   [B, pred_len, c_out]
    emb: [B, hidden]
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", self.enc_in)

        hidden = getattr(configs, "d_model", 128)
        n_layers = getattr(configs, "e_layers", 2)
        dropout = getattr(configs, "dropout", 0.0)

        # Linears adaptables para PROCEED
        self.in_proj = nn.Linear(self.enc_in, hidden)

        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        mlp_hidden = getattr(configs, "mlp_hidden", hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Linear(hidden, self.pred_len * self.c_out)

    def forward(self, x_enc, x_mark_enc=None, return_emb=False, *args, **kwargs):
        x = self.in_proj(x_enc)      # [B, L, H]
        out, _ = self.lstm(x)        # [B, L, H]
        emb = out[:, -1, :]          # [B, H]

        h = self.mlp(emb)            # [B, H]
        y = self.out_proj(h).view(x_enc.size(0), self.pred_len, self.c_out)

        return (y, emb) if return_emb else y
