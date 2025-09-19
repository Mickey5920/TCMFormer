# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# ======= Utilities =======
# =========================
def maybe_compile(module: nn.Module, enable_compile: bool = True):
    if enable_compile and hasattr(torch, "compile"):
        try:
            module = torch.compile(module, fullgraph=False, dynamic=True)
        except Exception:
            pass
    return module

def rope_rotate(x, cos, sin):
    """
    x:   [B, T, h, hd] （hd 为偶数）
    cos: [1, T, 1, hd/2]
    sin: [1, T, 1, hd/2]
    对最后一维 (even, odd) 成对应用旋转。
    """
    x1, x2 = x[..., ::2], x[..., 1::2]  # [B, T, h, hd/2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)  # [B, T, h, hd]

def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    """
    Return cos, sin with shape [1, T, 1, hd/2]
    (注意：与 x[..., ::2]/x[..., 1::2] 的尺寸一致，不要重复成 hd)
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()              # [T]
    freqs = torch.einsum('t,d->td', t, inv_freq)                  # [T, half]
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)              # [1, T, 1, half]
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)              # [1, T, 1, half]
    return cos, sin


# ====================================================
# ======== Channel Attention (kept & improved) =======
# ====================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        mid = max(8, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, mid)
        self.fc2 = nn.Linear(mid, in_channels)

    def forward(self, x):
        # x: [B, C] or [B, N, C]
        if x.dim() == 3:
            pooled = x.mean(1)  # [B, C]
        else:
            pooled = x
        attn = F.relu(self.fc1(pooled))
        attn = torch.sigmoid(self.fc2(attn))  # [B, C]
        if x.dim() == 3:
            attn = attn.unsqueeze(1)
        return x * attn

# ================================================
# ======= Capsule for Tabular (vectorized) =======
# ================================================
class PrimaryCapsules(nn.Module):
    """
    Map each tabular group to multiple primary capsules in parallel.
    """
    def __init__(self, in_dim, num_caps=4, cap_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.proj = nn.Linear(in_dim, num_caps * cap_dim)

    def forward(self, x):  # x: [B, in_dim]
        B = x.size(0)
        out = self.proj(x).view(B, self.num_caps, self.cap_dim)  # [B, C1, D1]
        # squash
        norm = out.norm(dim=-1, keepdim=True) + 1e-8
        out = (out / norm) * (norm**2 / (1.0 + norm**2))
        return out  # [B, C1, D1]

class DigitCapsules(nn.Module):
    """
    One shared DigitCaps layer that aggregates all primary capsules (concatenated from groups)
    to a single vector representation via dynamic routing.
    """
    def __init__(self, in_caps, in_dim, out_caps=1, out_dim=256, routing_iters=3):
        super().__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        # Transform matrices
        self.W = nn.Parameter(0.02 * torch.randn(1, in_caps, out_caps, out_dim, in_dim))

    def forward(self, u):  # u: [B, in_caps, in_dim]
        B = u.size(0)
        # W: [1, in_caps, out_caps, out_dim, in_dim] -> [B, in_caps, out_caps, out_dim, in_dim]
        W = self.W.expand(B, -1, -1, -1, -1)
        # u: [B, in_caps, in_dim] -> [B, in_caps, 1, in_dim] -> [B, in_caps, 1, in_dim, 1]
        u = u.unsqueeze(2).unsqueeze(-1)
        # matmul: [B, in_caps, out_caps, out_dim, in_dim] @ [B, in_caps, 1, in_dim, 1]
        u_hat = torch.matmul(W, u).squeeze(-1)  # [B, in_caps, out_caps, out_dim]

        b = torch.zeros(B, self.in_caps, self.out_caps, device=u_hat.device)
        for _ in range(self.routing_iters):
            c = torch.softmax(b, dim=2).unsqueeze(-1)      # [B, in_caps, out_caps, 1]
            s = (c * u_hat).sum(dim=1)                     # [B, out_caps, out_dim]
            # squash
            s_norm = s.norm(dim=-1, keepdim=True) + 1e-8
            v = (s / s_norm) * (s_norm**2 / (1.0 + s_norm**2))  # [B, out_caps, out_dim]
            b = b + (u_hat * v.unsqueeze(1)).sum(-1)       # [B, in_caps, out_caps]

        v = v.squeeze(1)  # [B, out_dim] if out_caps == 1
        return v

class TabularCapsuleExtractor(nn.Module):
    """
    Take multiple tabular groups; build primary caps per group, then concatenate and route.
    """
    def __init__(self, dims, out_dim=256, num_caps_per_group=4, cap_dim=16, routing_iters=3):
        super().__init__()
        self.groups = nn.ModuleList([
            PrimaryCapsules(d, num_caps=num_caps_per_group, cap_dim=cap_dim)
            for d in dims
        ])
        in_caps = len(dims) * num_caps_per_group
        self.digit_caps = DigitCapsules(in_caps=in_caps, in_dim=cap_dim,
                                        out_caps=1, out_dim=out_dim, routing_iters=routing_iters)

    def forward(self, features):  # list[tensor], each [B, d]
        prims = [g(f) for g, f in zip(self.groups, features)]  # each [B, C1, D1]
        u = torch.cat(prims, dim=1)  # [B, in_caps, cap_dim]
        v = self.digit_caps(u)       # [B, out_dim]
        return v



class SDPAMultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads=8, rope=False, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n = n_heads
        self.hdim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = rope
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, rope_cache=None):  # x: [B,T,D]
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n, self.hdim).transpose(2, 3)  # [B, T, h, 3, hd]
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]             # [B, T, h, hd]

        # ✅ 先在 [B, T, h, hd] 形状下应用 RoPE（cos/sin: [1, T, 1, hd/2]）
        if self.rope and rope_cache is not None:
            cos, sin = rope_cache  # [1, L, 1, hd/2]
            assert self.hdim % 2 == 0, "head_dim must be even for RoPE"
            q = rope_rotate(q, cos[:, :T], sin[:, :T])  # 现在广播维度完全对齐
            k = rope_rotate(k, cos[:, :T], sin[:, :T])

        # 再转成 [B, h, T, hd] 进入 SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # SDPA
        q_ = q.reshape(B * self.n, T, self.hdim)
        k_ = k.reshape(B * self.n, T, self.hdim)
        v_ = v.reshape(B * self.n, T, self.hdim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False,
        )  # [B*h, T, hd]
        attn = attn.reshape(B, self.n, T, self.hdim).transpose(1, 2).reshape(B, T, D)
        attn = self.out(attn)
        return self.dropout(attn)

# ====================================================
# ======= Fast Transformer Blocks (SDPA/RMSNorm) =====
# ====================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return (x / (norm + self.eps)) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult)
        self.w1 = nn.Linear(dim, inner, bias=True)
        self.w2 = nn.Linear(dim, inner, bias=True)
        self.w3 = nn.Linear(inner, dim, bias=True)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0, rope=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SDPAMultiheadAttention(dim, n_heads=n_heads, rope=rope, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, mult=4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope_cache=None):
        x = x + self.drop(self.attn(self.norm1(x), rope_cache=rope_cache))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """
    Shared encoder for 4 modalities with a modality embedding + CLS token.
    """
    def __init__(self, in_dim=64, model_dim=256, depth=6, n_heads=8, rope=True, dropout=0.0, use_checkpoint=False, max_len=512, n_modalities=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, model_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.mod_emb = nn.Embedding(n_modalities, model_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, n_heads=n_heads, dropout=dropout, rope=rope)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(model_dim)
        self.use_checkpoint = use_checkpoint
        self.rope = rope
        self.max_len = max_len
        self.rope_cache = None  # lazily built

    def _rope(self, x):
        B, T, D = x.shape
        # 取第一个 block 的注意力头数
        n_heads = self.blocks[0].attn.n
        head_dim = D // n_heads
        if (self.rope_cache is None) or (self.rope_cache[0].size(1) < T):
            self.rope_cache = build_rope_cache(max(T, self.max_len), head_dim, x.device)
        return self.rope_cache

    def forward(self, x, modality_id: int):
        """
        x: [B, T, in_dim] (padded if needed). No mask assumed here.
        """
        B, T, _ = x.shape
        x = self.in_proj(x)  # [B,T,D]
        # prepend CLS
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,1+T,D]
        # add modality embedding broadcast
        x = x + self.mod_emb.weight[modality_id].view(1, 1, -1)

        rope_cache = self._rope(x) if self.rope else None
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, rope_cache, use_reentrant=False)
            else:
                x = blk(x, rope_cache=rope_cache)
        x = self.norm(x)  # [B,1+T,D]
        cls_out = x[:, 0]  # [B,D]
        return cls_out

# ======================================================
# ======= Multi-Sequence Fusion (simple, effective) =====
# ======================================================
class MultiSequenceFuser(nn.Module):
    def __init__(self, n_modal=4, feat_dim=256):
        super().__init__()
        self.attn = ChannelAttention(n_modal * feat_dim)
        self.fc = nn.Linear(n_modal * feat_dim, feat_dim)

    def forward(self, seq_feats):  # list of [B,D]
        x = torch.cat(seq_feats, dim=1)  # [B, n_modal*D]
        x = self.attn(x)
        x = F.relu(self.fc(x))
        return x

# ======================================================
# =================== Full Model =======================
# ======================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x1, x2):  # [B,D], [B,D]
        q = self.fc_q(x1).unsqueeze(1)  # [B,1,D]
        k = self.fc_k(x2).unsqueeze(1)  # [B,1,D]
        v = self.fc_v(x2).unsqueeze(1)
        scale = math.sqrt(x1.size(1))
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / scale, dim=-1)
        out = (attn * v).sum(1)  # [B,D]
        out = self.fc_out(out + x1 + x2)
        return out


class CapsuleTransformerMultiDiseaseRiskModel(nn.Module):
    """
    - Tabular: Capsule extractor
    - Event sequences: Shared multi-layer Transformer encoder with modality embeddings
    - Same forward signature & logits as original
    """
    def __init__(
        self,
        tab_dims,
        event_feat_dim=64,
        out_dim=947,
        hidden_dim=256,
        # Transformer hyperparams
        tr_depth=6,
        tr_heads=8,
        tr_rope=True,
        tr_dropout=0.0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        # 1) Tabular via Capsules
        self.tabular_extractor = TabularCapsuleExtractor(
            tab_dims, out_dim=hidden_dim, num_caps_per_group=4, cap_dim=16, routing_iters=3
        )

        # 2) Shared Transformer for sequences (4 modalities)
        self.seq_encoder = TransformerEncoder(
            in_dim=event_feat_dim, model_dim=hidden_dim, depth=tr_depth, n_heads=tr_heads,
            rope=tr_rope, dropout=tr_dropout, use_checkpoint=gradient_checkpointing,
            n_modalities=4
        )

        # 3) Fuse 4 modalities
        self.fuser = MultiSequenceFuser(n_modal=4, feat_dim=hidden_dim)

        # 4) Cross fusion X1 + X2
        self.cross_attn = CrossAttentionFusion(dim=hidden_dim)

        # 5) Output
        self.out_fc = nn.Linear(hidden_dim, out_dim)

    @torch.inference_mode(False)
    def forward(
        self, Demo, Physical, Biomarkers, Lifestyle, Mental, Environmental, Genetic, Other,
        X_before_phecode, X_before_phenotypes, X_before_opcs4, X_before_drug
    ):
        # 1) Tabular features -> capsule vector
        feats_list = [Demo, Physical, Biomarkers, Lifestyle, Mental, Environmental, Genetic, Other]
        x1 = self.tabular_extractor(feats_list)  # [B, hidden_dim]

        # 2) Each event sequence -> Transformer vector (shared encoder + modality id)
        seq_feats = []
        seq_inputs = [
            (X_before_phecode, 0),
            (X_before_phenotypes, 1),
            (X_before_opcs4, 2),
            (X_before_drug, 3),
        ]
        for seq, mid in seq_inputs:
            # seq: [B, T, C] where C == event_feat_dim
            feat = self.seq_encoder(seq, modality_id=mid)  # [B, hidden_dim]
            seq_feats.append(feat)
        x2 = self.fuser(seq_feats)  # [B, hidden_dim]

        # 3) Cross fusion
        fused = self.cross_attn(x1, x2)  # [B, hidden_dim]

        # 4) logits
        logits = self.out_fc(fused)
        return logits

# ================================
# ===== Convenience builder ======
# ================================
def build_model(
    tab_dims,
    event_feat_dim=64,
    out_dim=947,
    hidden_dim=256,
    tr_depth=6,
    tr_heads=8,
    tr_rope=True,
    tr_dropout=0.0,
    gradient_checkpointing=False,
    enable_compile=True,
):
    model = CapsuleTransformerMultiDiseaseRiskModel(
        tab_dims=tab_dims,
        event_feat_dim=event_feat_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        tr_depth=tr_depth,
        tr_heads=tr_heads,
        tr_rope=tr_rope,
        tr_dropout=tr_dropout,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = maybe_compile(model, enable_compile)
    return model


