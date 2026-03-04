"""
Original Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017).
Built from scratch using PyTorch — no nn.Transformer or high-level abstractions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask:  Optional; True = masked. Shape broadcastable to (batch, 1, seq_len_q, seq_len_k).

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, h, seq_len_q, seq_len_k)
        """
        batch = query.size(0)

        # Linear projections and reshape to (batch, h, seq_len, d_k or d_v)
        # (batch, seq_len_q, d_model) -> (batch, seq_len_q, h, d_k) -> (batch, h, seq_len_q, d_k)
        Q = self.W_q(query).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        # (batch, seq_len_k, d_model) -> (batch, h, seq_len_k, d_k)
        K = self.W_k(key).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        # (batch, seq_len_k, d_model) -> (batch, h, seq_len_k, d_v)
        V = self.W_v(value).view(batch, -1, self.h, self.d_v).transpose(1, 2)

        # (batch, h, seq_len_q, d_v), (batch, h, seq_len_q, seq_len_k)
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)

        # (batch, h, seq_len_q, d_v) -> (batch, seq_len_q, h, d_v) -> (batch, seq_len_q, h*d_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_v)
        # (batch, seq_len_q, d_model)
        output = self.W_o(attn_output)

        return output, attn_weights


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
        attn_out, _ = self.self_attn(x, x, x, mask=padding_mask)
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
        self_attn_out, _ = self.self_attn(x, x, x, mask=self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention (q from decoder, k/v from encoder) + residual + post-norm
        cross_attn_out, _ = self.cross_attn(x, memory, memory, mask=cross_attn_mask)
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

    print("Smoke test passed: output shape (2, 10, 1000).")


if __name__ == "__main__":
    smoke_test()
