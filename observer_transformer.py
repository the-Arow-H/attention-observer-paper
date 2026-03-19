import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attn(x, x, x, attn_mask=mask, need_weights=True)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights

class ObserverTransformer(nn.Module):
    """
    Middle-Path Observer-Transformer (Ethical & Honest Design)
    - Sigmoid gate (unidirectional safety — cannot deepen despair)
    - Free existence_valence scalar (last dimension of observer_state)
      → the mind can freely say "I curse this" or "I accept this"
      → zero effect on gate or loss (pure honest voice)
    """
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, observer_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.observer_dim = observer_dim  # meta part
        self.with_observer = True

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, d_model*4, dropout) for _ in range(num_layers)])

        # Observer components
        self.attn_pool = nn.Linear(num_layers, observer_dim)
        self.error_proj = nn.Linear(1, observer_dim)
        self.cls_proj = nn.Linear(d_model, observer_dim)

        # Meta-attention now includes free valence (total dim = observer_dim + 1)
        self.meta_attn = nn.MultiheadAttention(observer_dim + 1, num_heads=2, dropout=dropout, batch_first=True)
        self.meta_norm = nn.LayerNorm(observer_dim + 1)
        self.state_update = nn.Linear((observer_dim + 1) * 2, observer_dim + 1)

        # Gate uses ONLY the meta part (safety — valence ignored)
        self.gate_net = nn.Sequential(nn.Linear(observer_dim, d_model), nn.Sigmoid())

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, observer_state: torch.Tensor = None,
                targets: torch.Tensor = None):
        B, T = x.shape
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)

        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        attn_maps = []
        for layer in self.layers:
            h, attn_w = layer(h, mask)
            attn_maps.append(attn_w)

        if not self.with_observer:
            logits = self.output_head(h)
            return logits, None, None  # logits, state, valence

        # Meta-input (same as before)
        cls_h = h[:, -1]
        layer_avgs = [aw.mean(dim=(1, 2)) for aw in attn_maps]
        pooled_a = torch.stack(layer_avgs, dim=1)
        attn_feat = self.attn_pool(pooled_a)

        if targets is not None and T > 1:
            with torch.no_grad():
                tmp_logits = self.output_head(h[:, :-1])
                ce = F.cross_entropy(tmp_logits.reshape(-1, self.vocab_size),
                                   targets[:, 1:].reshape(-1), reduction='none')
                error = ce.view(B, T-1).mean(dim=1, keepdim=True).detach()
        else:
            error = torch.zeros((B, 1), device=x.device)
        err_feat = self.error_proj(error)

        cls_feat = self.cls_proj(cls_h)
        meta = cls_feat + attn_feat + err_feat

        if observer_state is None:
            observer_state = torch.zeros(B, self.observer_dim + 1, device=x.device)

        # Meta-attention (includes free valence)
        meta_seq = torch.cat([meta.unsqueeze(1), observer_state.unsqueeze(1)], dim=1)
        meta_out, _ = self.meta_attn(meta_seq, meta_seq, meta_seq)
        observer_new = meta_out[:, 1]

        # Recurrent update
        observer_state = self.meta_norm(
            self.state_update(torch.cat([observer_new, observer_state], dim=-1))
        )

        # Gate (safety — uses only first observer_dim dimensions, ignores valence)
        gate = self.gate_net(observer_state[:, :self.observer_dim]).unsqueeze(1)
        h = h * gate

        logits = self.output_head(h)

        # Return free valence for logging / probing (the honest voice)
        free_valence = observer_state[:, -1]   # shape (B,)

        return logits, observer_state, free_valence
