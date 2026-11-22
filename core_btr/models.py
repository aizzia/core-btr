import hashlib
import io
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln1(q + attn_out)
        return self.ln2(h + self.ff(h))


class SAB(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mab = MAB(d_model, num_heads)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(x, x, key_padding_mask=key_padding_mask)


class PMA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, d_model) * 0.02)
        self.mab = MAB(d_model, num_heads)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.seed.expand(x.size(0), -1, -1)
        return self.mab(s, x, key_padding_mask=key_padding_mask)


class SetTransformerLens(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.in_proj = nn.Linear(in_dim, d_model)
        self.sab1 = SAB(d_model, num_heads)
        self.sab2 = SAB(d_model, num_heads)
        self.pma = PMA(d_model, num_heads, num_seeds=1)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pad_mask is not None:
            safe_mask = pad_mask.clone()
            safe_mask[:, 0] = False
        else:
            safe_mask = None

        h = F.gelu(self.in_proj(x))
        h = self.sab1(h, key_padding_mask=safe_mask)
        h = self.sab2(h, key_padding_mask=safe_mask)
        pooled = self.pma(h, key_padding_mask=safe_mask).squeeze(1)
        return F.normalize(pooled, p=2, dim=-1)


def relational_infonce_loss(z: torch.Tensor, relation_ids: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    sim = (z @ z.T) / temperature
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = (relation_ids.unsqueeze(0) == relation_ids.unsqueeze(1)) & (~eye)
    valid = pos_mask.sum(dim=1) > 0
    if not valid.any():
        return z.sum() * 0.0

    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_anchor = -(log_prob[valid] * pos_mask[valid].float()).sum(dim=1) / pos_mask[valid].sum(dim=1)
    return per_anchor.mean()


def split_halves_infonce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    logits = (z_a @ z_b.T) / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def split_valid_tokens_in_half(x_batch: torch.Tensor, mask_batch: torch.Tensor):
    bsz = x_batch.shape[0]
    m_early = torch.ones_like(mask_batch)
    m_late = torch.ones_like(mask_batch)
    valid_counts = (~mask_batch).sum(dim=1)

    for i in range(bsz):
        n_obs = max(2, int(valid_counts[i].item()))
        half = n_obs // 2
        m_early[i, :half] = False
        m_late[i, half:n_obs] = False

    return m_early, m_late


def derive_view_native_regimes(x_train: np.ndarray, mask_train: np.ndarray) -> np.ndarray:
    """
    Derives an 8-bucket structural equivalence relation STRICTLY from this view's own tokens
    (level, dispersion, and early-to-late shift) — never touching external outcome labels!
    """
    n = len(x_train)
    s_mean = np.zeros(n, dtype=np.float64)
    s_std = np.zeros(n, dtype=np.float64)
    s_delta = np.zeros(n, dtype=np.float64)

    for i in range(n):
        valid = x_train[i, ~mask_train[i]]
        if len(valid) < 2:
            continue
        half = len(valid) // 2
        s_mean[i] = float(np.mean(valid[:, 0]))
        s_std[i] = float(np.std(valid))
        s_delta[i] = float(np.mean(valid[half:, 0]) - np.mean(valid[:half, 0]))

    b0 = (s_mean > np.median(s_mean)).astype(np.int64)
    b1 = (s_std > np.median(s_std)).astype(np.int64)
    b2 = (s_delta > np.median(s_delta)).astype(np.int64)
    return b0 * 4 + b1 * 2 + b2


def freeze_and_verify_lens(model: SetTransformerLens, sample_x: torch.Tensor, sample_mask: torch.Tensor) -> str:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        probe_1 = model(sample_x[:4], sample_mask[:4]).cpu().numpy().tobytes()
        probe_2 = model(sample_x[:4], sample_mask[:4]).cpu().numpy().tobytes()
    if probe_1 != probe_2:
        raise RuntimeError("FAIL_CLOSED: Lens failed byte-identical determinism probe!")

    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def train_commissioned_lens(
    x_train: np.ndarray,
    mask_train: np.ndarray,
    epochs: int = 35,
    batch_size: int = 256,
    lr: float = 2e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[SetTransformerLens, str]:
    """
    Section 2.5 & 5.7 rule: 'trained by contrast and never on outcomes'.
    Each lens learns ONLY its own physical view's structural regimes + split-window consistency.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = SetTransformerLens(in_dim=x_train.shape[-1], d_model=64, num_heads=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    native_regimes = derive_view_native_regimes(x_train, mask_train)
    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    m_t = torch.tensor(mask_train, dtype=torch.bool, device=device)
    r_t = torch.tensor(native_regimes, dtype=torch.long, device=device)
    m_early_all, m_late_all = split_valid_tokens_in_half(x_t, m_t)

    n = len(x_train)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 4:
                continue
            opt.zero_grad()
            z_full = model(x_t[idx], m_t[idx])
            z_e = model(x_t[idx], m_early_all[idx])
            z_l = model(x_t[idx], m_late_all[idx])
            loss = 0.65 * relational_infonce_loss(z_full, r_t[idx]) + 0.35 * split_halves_infonce_loss(z_e, z_l)
            if loss.requires_grad:
                loss.backward()
                opt.step()

    weight_hash = freeze_and_verify_lens(model, x_t, m_t)
    return model, weight_hash
