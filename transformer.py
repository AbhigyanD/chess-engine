"""
Original Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).
Built from scratch using PyTorch — no nn.Transformer or high-level abstractions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a padding mask for attention.

    Positions that are padded (index >= length) get True (masked out).
    Returns mask of shape (batch_size, 1, 1, max_len) for broadcasting
    in attention: True = masked (will get -inf added to scores).

    Args:
        lengths: (batch_size,) — actual sequence length per batch item.
        max_len: Maximum sequence length.
        device: Device for the mask tensor.

    Returns:
        Boolean mask (batch_size, 1, 1, max_len), True where padded.
    """
    batch_size = lengths.size(0)
    if device is None:
        device = lengths.device
    # (batch_size, max_len): True where position >= length (i.e. padded)
    mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    # (batch_size, 1, 1, max_len) for attention broadcast
    return mask.unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a causal (autoregressive) mask for decoder self-attention.

    Position i can attend only to positions 0..i (inclusive).
    Returns mask of shape (1, 1, seq_len, seq_len): True = masked (future positions).

    Args:
        seq_len: Sequence length.
        device: Device for the mask tensor.

    Returns:
        Boolean mask (1, 1, seq_len, seq_len), True where attention is forbidden.
    """
    if device is None:
        device = torch.device("cpu")
    # Upper triangular (excluding diagonal) = future positions
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters of a module.

    Args:
        model: PyTorch module.

    Returns:
        (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# -----------------------------------------------------------------------------
# Scaled Dot-Product Attention
# -----------------------------------------------------------------------------


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Supports optional attention mask (True = masked out; add -inf before softmax).
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, heads, seq_len_q, d_k)
            key:   (batch, heads, seq_len_k, d_k)
            value: (batch, heads, seq_len_k, d_v)
            mask:  Optional (batch, 1, seq_len_q, seq_len_k) or broadcastable; True = masked.

        Returns:
            output: (batch, heads, seq_len_q, d_v)
            attention_weights: (batch, heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        # (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask is True where we want to mask out (set to -inf)
            scores = scores.masked_fill(mask, float("-inf"))

        # (batch, heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # (batch, heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


# -----------------------------------------------------------------------------
# Multi-Head Attention
# -----------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: linear projections for Q/K/V, split into h heads,
    parallel scaled dot-product attention, concatenate and project.
    Supports self-attention (q=k=v from same source) and cross-attention (q from one, k/v from another).
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model == h * d_k, "d_model must equal h * d_k"
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_v)
        self.W_o = nn.Linear(h * d_v, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  Optional; True = masked. Shape broadcastable to (batch, 1, seq_len_q, seq_len_k).
            rope_cos, rope_sin: Optional (seq_len, d_k/2) for RoPE on Q and K.
            past_key_value: Optional (past_k, past_v) for KV cache.
            use_cache: If True, return updated (K, V) as third output.

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, h, seq_len_q, seq_len_k)
            past_kv: (K, V) if use_cache else None
        """
        batch = query.size(0)

        # Linear projections and reshape to (batch, h, seq_len, d_k or d_v)
        # (batch, seq_len_q, d_model) -> (batch, seq_len_q, h, d_k) -> (batch, h, seq_len_q, d_k)
        Q = self.W_q(query).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        # (batch, seq_len_k, d_model) -> (batch, h, seq_len_k, d_k)
        K = self.W_k(key).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        # (batch, seq_len_k, d_model) -> (batch, h, seq_len_k, d_v)
        V = self.W_v(value).view(batch, -1, self.h, self.d_v).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            Q, K = apply_rotary_emb(Q, K, rope_cos, rope_sin)

        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        if use_cache:
            past_kv = (K.detach(), V.detach())

        # (batch, h, seq_len_q, d_v), (batch, h, seq_len_q, seq_len_k)
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)

        # (batch, h, seq_len_q, d_v) -> (batch, seq_len_q, h, d_v) -> (batch, seq_len_q, h*d_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_v)
        # (batch, seq_len_q, d_model)
        output = self.W_o(attn_output)

        return output, attn_weights, past_kv


# -----------------------------------------------------------------------------
# RMSNorm (LLaMA-style)
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root mean square layer normalization (no mean subtraction). Used in LLaMA."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# -----------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# -----------------------------------------------------------------------------


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise FFN: two linear layers with ReLU in between.
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2. Inner dimension d_ff.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(x)


class PositionwiseFeedForwardGELU(nn.Module):
    """
    FFN with GELU activation (GPT-2 / LLaMA style).
    FFN(x) = (gelu(x W_1) * (x V)) W_2 for SwiGLU; or gelu(x W_1) W_2 for plain GELU.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: silu(w1(x)) * w3(x) then w2
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        x = F.gelu(self.linear1(x))
        return self.dropout(self.linear2(x))


# -----------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# -----------------------------------------------------------------------------


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to Q and K. cos, sin shape (seq_len, d_k/2).
    q, k shape (batch, heads, seq_len, d_k). Modifies in place and returns (q, k).
    """
    # (batch, heads, seq_len, d_k) -> (batch, heads, seq_len, d_k/2, 2)
    d = q.size(-1) // 2
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    # cos, sin (seq_len, d) -> (1, 1, seq_len, d)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE). Precomputes cos/sin for positions 0..max_len-1."""

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(
        self,
        seq_len: int,
        start: int = 0,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin for positions [start, start+seq_len). For KV cache use start=past_len."""
        t = torch.arange(
            start, start + seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos(), freqs.sin()


# -----------------------------------------------------------------------------
# Positional Encoding (sinusoidal, fixed)
# -----------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (fixed, not learned).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input. Input must have seq_len <= max_len used in __init__.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# -----------------------------------------------------------------------------
# Encoder Layer (post-norm: LayerNorm(x + Sublayer(x)))
# -----------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    """
    One encoder layer: multi-head self-attention + FFN, each with residual connection and layer norm (post-norm).
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            padding_mask: Optional (batch, 1, 1, seq_len); True = padded position.

        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual and post-norm
        # (batch, seq_len, d_model)
        attn_out, _, _ = self.self_attn(x, x, x, mask=padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual and post-norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


# -----------------------------------------------------------------------------
# Decoder Layer
# -----------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """
    One decoder layer: masked multi-head self-attention + cross-attention over encoder output + FFN,
    each with residual connection and layer norm (post-norm).
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, tgt_len, d_model).
            memory: Encoder output (batch, src_len, d_model).
            self_attn_mask: Causal + optional padding for self-attn (batch, 1, tgt_len, tgt_len).
            cross_attn_mask: Optional mask for cross-attn (batch, 1, 1, src_len) to ignore padding in memory.

        Returns:
            (batch, tgt_len, d_model)
        """
        # Masked self-attention + residual + post-norm
        # (batch, tgt_len, d_model)
        self_attn_out, _, _ = self.self_attn(x, x, x, mask=self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention (q from decoder, k/v from encoder) + residual + post-norm
        cross_attn_out, _, _ = self.cross_attn(x, memory, memory, mask=cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # FFN + residual + post-norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    Transformer encoder: embedding + positional encoding + stack of N encoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        N: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, h, d_k, d_v, d_ff, dropout=dropout)
                for _ in range(N)
            ]
        )
        self.d_model = d_model

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) — source token indices.
            src_lengths: Optional (batch,) — actual lengths for padding mask.

        Returns:
            (batch, src_len, d_model)
        """
        # (batch, src_len, d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        padding_mask = None
        if src_lengths is not None:
            padding_mask = create_padding_mask(src_lengths, src.size(1), device=src.device)

        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        return x


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------


class Decoder(nn.Module):
    """
    Transformer decoder: embedding + positional encoding + stack of N decoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        N: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, h, d_k, d_v, d_ff, dropout=dropout)
                for _ in range(N)
            ]
        )
        self.d_model = d_model

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_lengths: Optional[torch.Tensor] = None,
        memory_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch, tgt_len) — target token indices.
            memory: (batch, src_len, d_model) — encoder output.
            tgt_lengths: Optional (batch,) — target lengths for padding + causal.
            memory_lengths: Optional (batch,) — source lengths for cross-attn padding mask.

        Returns:
            (batch, tgt_len, d_model)
        """
        tgt_len = tgt.size(1)
        # (batch, tgt_len, d_model)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask for self-attention: (1, 1, tgt_len, tgt_len)
        self_attn_mask = create_causal_mask(tgt_len, device=tgt.device)

        # Optionally combine with padding mask for target (don't attend to padding)
        if tgt_lengths is not None:
            padding_mask_tgt = create_padding_mask(tgt_lengths, tgt_len, device=tgt.device)
            # Combine: mask out if causal OR padding. (batch, 1, 1, tgt_len) broadcast with (1,1,tgt_len,tgt_len)
            # We need (batch, 1, tgt_len, tgt_len): for each query position, mask out key positions that are padded
            # padding_mask_tgt is (batch, 1, 1, tgt_len) — True where key is padded. Broadcast to (batch, 1, tgt_len, tgt_len)
            self_attn_mask = self_attn_mask | padding_mask_tgt

        # Cross-attention mask: mask encoder padding. (batch, 1, 1, src_len)
        cross_attn_mask = None
        if memory_lengths is not None:
            cross_attn_mask = create_padding_mask(
                memory_lengths, memory.size(1), device=memory.device
            )

        for layer in self.layers:
            x = layer(
                x,
                memory,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )

        return x


# -----------------------------------------------------------------------------
# Full Transformer
# -----------------------------------------------------------------------------


class Transformer(nn.Module):
    """
    Full Transformer: encoder + decoder + final linear projection.
    Output logits over target vocabulary; softmax can be applied in loss (e.g. CrossEntropyLoss).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert d_model == h * d_k, "d_model must equal h * d_k"
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            N=N,
            h=h,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            N=N,
            h=h,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.final_proj = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) — source token indices.
            tgt: (batch, tgt_len) — target token indices (e.g. right-shifted for teacher forcing).
            src_lengths: Optional (batch,) — source lengths.
            tgt_lengths: Optional (batch,) — target lengths.

        Returns:
            Logits (batch, tgt_len, tgt_vocab_size). Apply softmax for token prediction.
        """
        # (batch, src_len, d_model)
        memory = self.encoder(src, src_lengths=src_lengths)
        # (batch, tgt_len, d_model)
        decoder_out = self.decoder(
            tgt,
            memory,
            tgt_lengths=tgt_lengths,
            memory_lengths=src_lengths,
        )
        # (batch, tgt_len, tgt_vocab_size)
        logits = self.final_proj(decoder_out)
        return logits


# -----------------------------------------------------------------------------
# Decoder-only LLM (GPT-style)
# -----------------------------------------------------------------------------


class DecoderOnlyLM(nn.Module):
    """
    Decoder-only transformer for causal language modeling (GPT-style LLM).
    Single stack of layers: masked self-attention + FFN, no encoder or cross-attention.
    Use for next-token prediction and autoregressive generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert d_model == h * d_k, "d_model must equal h * d_k"
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        # Reuse EncoderLayer: same structure (self-attn + FFN); we pass causal mask as padding_mask
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, h, d_k, d_v, d_ff, dropout=dropout)
                for _ in range(N)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Causal forward pass: predict logits for each position given previous tokens.

        Args:
            input_ids: (batch, seq_len) — token indices.
            lengths: Optional (batch,) — actual lengths for padding mask.

        Returns:
            Logits (batch, seq_len, vocab_size). Use for next-token loss: logits[:, :-1] vs input_ids[:, 1:].
        """
        seq_len = input_ids.size(1)
        # (batch, seq_len, d_model)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal + optional padding mask for self-attention
        # (1, 1, seq_len, seq_len)
        mask = create_causal_mask(seq_len, device=input_ids.device)
        if lengths is not None:
            # (batch, 1, 1, seq_len) -> (batch, 1, seq_len, seq_len)
            pad_mask = create_padding_mask(lengths, seq_len, device=input_ids.device)
            pad_mask = pad_mask.expand(-1, -1, seq_len, -1)
            mask = mask | pad_mask

        for layer in self.layers:
            x = layer(x, padding_mask=mask)

        # (batch, seq_len, vocab_size)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation. Extends input_ids by sampling (or greedy) next tokens.

        Args:
            input_ids: (batch, seq_len) — prompt token indices.
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: If set, stop when this token is generated.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, sample only from top-k logits (None = no cutoff).
            do_sample: If False, use greedy (argmax) decoding.

        Returns:
            (batch, seq_len + num_generated) — extended token ids.
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = next(self.parameters()).device
        generated = input_ids
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Forward on current sequence (causal: only last position matters for next token)
            logits = self.forward(generated, lengths=None)
            # (batch, vocab_size)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1].unsqueeze(-1)] = float("-inf")

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                eos_reached = eos_reached | (next_token.squeeze(-1) == eos_token_id)
                if eos_reached.all():
                    break

        return generated


# -----------------------------------------------------------------------------
# Modern LLM (Claude / GPT / LLaMA style): pre-norm, RMSNorm, GELU, RoPE, KV cache
# -----------------------------------------------------------------------------


class LLMLayer(nn.Module):
    """
    Single decoder-only layer with pre-norm (RMSNorm), self-attention with RoPE, and GELU FFN.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        dropout: float = 0.1,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForwardGELU(d_model, d_ff, dropout=dropout, use_swiglu=use_swiglu)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + self-attention + residual
        # (batch, seq_len, d_model)
        x_norm = self.attn_norm(x)
        attn_out, _, past_kv = self.self_attn(
            x_norm, x_norm, x_norm,
            mask=mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + self.dropout(attn_out)
        # Pre-norm + FFN + residual
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, past_kv


class LLM(nn.Module):
    """
    Modern decoder-only LLM: pre-norm, RMSNorm, GELU (or SwiGLU), RoPE, optional KV cache.
    Similar in spirit to LLaMA / GPT-2 / Claude-style models.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        max_len: int = 8192,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        assert d_model == h * d_k
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.rope = RotaryEmbedding(dim=d_k, max_len=max_len)
        self.layers = nn.ModuleList([
            LLMLayer(d_model, h, d_k, d_v, d_ff, dropout=dropout, use_swiglu=use_swiglu)
            for _ in range(N)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Returns logits and optionally updated key-value cache for generation.
        """
        seq_len = input_ids.size(1)
        past_len = past_key_values[0][0].size(2) if (past_key_values is not None and len(past_key_values) > 0) else 0
        # (batch, seq_len, d_model) — no scale by sqrt(d_model) for modern LLMs
        x = self.embedding(input_ids)

        rope_cos, rope_sin = self.rope(seq_len, start=past_len, device=x.device)

        mask = None
        if past_len == 0:
            mask = create_causal_mask(seq_len, device=input_ids.device)
            if lengths is not None:
                pad_mask = create_padding_mask(lengths, seq_len, device=input_ids.device)
                pad_mask = pad_mask.expand(-1, -1, seq_len, -1)
                mask = mask | pad_mask
        # When using cache we only have one new token; causal is handled by cache

        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, pkv = layer(x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin,
                            past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                present_key_values.append(pkv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache for efficiency."""
        self.eval()
        batch_size = input_ids.size(0)
        device = next(self.parameters()).device
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        generated = input_ids
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            use_cache = past_key_values is not None
            # When we have cache, we only pass the last token
            if use_cache:
                ids = generated[:, -1:]
            else:
                ids = generated
            logits, past_key_values = self.forward(ids, past_key_values=past_key_values, use_cache=True)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1].unsqueeze(-1)] = float("-inf")

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                eos_reached = eos_reached | (next_token.squeeze(-1) == eos_token_id)
                if eos_reached.all():
                    break

        return generated


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


def smoke_test() -> None:
    """Instantiate model with base config, run one forward pass, assert output shapes."""
    # Base model config from paper
    d_model = 512
    h = 8
    d_k = d_v = 64
    d_ff = 2048
    N = 6
    dropout = 0.1
    batch_size = 2
    src_len = 10
    tgt_len = 10
    src_vocab_size = 1000
    tgt_vocab_size = 1000

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        N=N,
        h=h,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        dropout=dropout,
    )

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    model.eval()
    with torch.no_grad():
        src = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
        logits = model(src, tgt)

    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size), (
        f"Expected logits shape (batch={batch_size}, tgt_len={tgt_len}, tgt_vocab={tgt_vocab_size}), "
        f"got {logits.shape}"
    )

    # Optional: with length masks
    src_lengths = torch.tensor([7, 10])
    tgt_lengths = torch.tensor([8, 10])
    with torch.no_grad():
        logits_masked = model(src, tgt, src_lengths=src_lengths, tgt_lengths=tgt_lengths)
    assert logits_masked.shape == (batch_size, tgt_len, tgt_vocab_size)

    # Decoder-only LLM smoke test
    lm = DecoderOnlyLM(
        vocab_size=1000,
        d_model=512,
        N=6,
        h=8,
        d_k=64,
        d_v=64,
        d_ff=2048,
        dropout=0.1,
    )
    lm.eval()
    with torch.no_grad():
        ids = torch.randint(0, 1000, (2, 10))
        logits_lm = lm(ids)
    assert logits_lm.shape == (2, 10, 1000)
    out = lm.generate(ids, max_new_tokens=5, do_sample=False)
    assert out.shape == (2, 15)

    # Modern LLM (RoPE, pre-norm, KV cache)
    modern = LLM(
        vocab_size=1000,
        d_model=256,
        N=2,
        h=4,
        d_k=64,
        d_v=64,
        d_ff=512,
        max_len=128,
        dropout=0.1,
    )
    modern.eval()
    with torch.no_grad():
        logits_m, _ = modern(ids, use_cache=False)
    assert logits_m.shape == (2, 10, 1000)
    out_m = modern.generate(ids, max_new_tokens=5, do_sample=False)
    assert out_m.shape == (2, 15)

    print("Smoke test passed: encoder-decoder; DecoderOnlyLM; LLM (RoPE, KV cache).")


if __name__ == "__main__":
    smoke_test()
