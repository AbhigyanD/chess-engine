"""
Train the modern LLM (Claude / GPT / LLaMA style): pre-norm, RMSNorm, GELU, RoPE, KV cache.
Includes a minimal BPE tokenizer and LR warmup + cosine decay.
Run with: python train_llm.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from transformer import LLM, count_parameters


# -----------------------------------------------------------------------------
# Minimal BPE tokenizer (no external deps)
# -----------------------------------------------------------------------------


class SimpleBPE:
    """
    Minimal byte-pair encoding tokenizer. Learns merges from a corpus.
    Similar in spirit to GPT-2/LLaMA tokenizers; vocab = base chars + merged pairs.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        pad_token: str = "<|pad|>",
        eos_token: str = "<|endoftext|>",
    ) -> None:
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[int, str] = {}
        self.inverse_vocab: dict[str, int] = {}
        self.pad_id: Optional[int] = None
        self.eos_id: Optional[int] = None

    def _get_pairs(self, word: list[str]) -> Counter:
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    def fit(self, text: str) -> None:
        """Learn BPE merges from text."""
        words = re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+", text)
        if not words:
            words = list(text)
        else:
            words = [list(w) for w in words]
        # Base vocab: all unique characters + special
        chars = set()
        for w in words:
            chars.update(w)
        special = [self.pad_token, self.eos_token]
        self.vocab = {i: t for i, t in enumerate(special + sorted(chars))}
        self.inverse_vocab = {t: i for i, t in self.vocab.items()}
        self.pad_id = 0
        self.eos_id = 1
        idx = len(self.vocab)
        self.merges = []
        while idx < self.vocab_size:
            pairs = Counter()
            for w in words:
                pairs.update(self._get_pairs(w))
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            new_tok = a + b
            self.merges.append((a, b))
            self.vocab[idx] = new_tok
            self.inverse_vocab[new_tok] = idx
            idx += 1
            new_words = []
            for w in words:
                new_w, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == (a, b):
                        new_w.append(new_tok)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                new_words.append(new_w)
            words = new_words
        self.vocab_size = len(self.vocab)

    def _tokenize(self, word: str) -> list[str]:
        if word not in self.inverse_vocab:
            tokens = list(word)
        else:
            tokens = [word]
        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode text to token ids."""
        words = re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+", text)
        if not words:
            words = list(text)
        ids = []
        for w in words:
            for t in self._tokenize(w):
                ids.append(self.inverse_vocab.get(t, self.pad_id or 0))
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids to string."""
        special = set()
        if self.pad_id is not None:
            special.add(self.pad_id)
        if self.eos_id is not None:
            special.add(self.eos_id)
        out = []
        for i in ids:
            if skip_special and i in special:
                continue
            out.append(self.vocab.get(i, ""))
        return "".join(out)


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------


def causal_lm_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """Next-token prediction loss. logits (B, T, V), target_ids (B, T+1)."""
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = target_ids[:, 1:].reshape(-1)
    ignore = pad_id if pad_id is not None else -100
    return F.cross_entropy(logits_flat, targets_flat, reduction="mean", ignore_index=ignore)


def get_lr_cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Learning rate with linear warmup and cosine decay to 10% of base."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.1 + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_llm(
    model: nn.Module,
    train_ids: torch.Tensor,
    num_steps: int = 200,
    batch_size: int = 8,
    seq_len: int = 64,
    lr: float = 3e-4,
    warmup_steps: int = 20,
    pad_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_cache: bool = False,
) -> list[float]:
    """Train with causal LM loss; optional LR warmup + cosine decay."""
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    n = train_ids.size(0)
    for step in range(num_steps):
        current_lr = get_lr_cosine_with_warmup(step, warmup_steps, num_steps, lr)
        for g in optimizer.param_groups:
            g["lr"] = current_lr
        starts = torch.randint(0, max(1, n - seq_len - 1), (batch_size,), device=device)
        batch_list = [
            train_ids[s.item() : s.item() + seq_len + 1]
            for s in starts
        ]
        batch = torch.stack(batch_list).to(device)
        input_ids = batch[:, :-1]
        try:
            out = model(input_ids, use_cache=use_cache)
        except TypeError:
            out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = causal_lm_loss(logits, batch, pad_id=pad_id)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 40 == 0:
            print(f"  step {step + 1}/{num_steps} loss {losses[-1]:.4f} lr {current_lr:.2e}")
    return losses


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Modern LLM: pre-norm, RMSNorm, GELU, RoPE, KV cache\n")

    # Corpus and BPE
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Large language models like GPT and LLaMA use transformer architectures. "
        "Attention is all you need. "
    ) * 80
    tokenizer = SimpleBPE(vocab_size=256, pad_token="<|pad|>", eos_token="<|endoftext|>")
    tokenizer.fit(corpus)
    train_ids = torch.tensor(tokenizer.encode(corpus, add_eos=False), dtype=torch.long)
    print(f"BPE vocab size: {tokenizer.vocab_size}, corpus tokens: {train_ids.size(0)}")

    # Modern LLM (small for fast demo)
    model = LLM(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        N=4,
        h=4,
        d_k=64,
        d_v=64,
        d_ff=512,
        max_len=256,
        dropout=0.1,
        padding_idx=tokenizer.pad_id,
        use_swiglu=False,
    ).to(device)
    total, trainable = count_parameters(model)
    print(f"Model parameters: {total:,} (trainable: {trainable:,})\n")

    print("Training (causal LM, warmup + cosine LR)...")
    train_llm(
        model,
        train_ids,
        num_steps=200,
        batch_size=16,
        seq_len=64,
        lr=3e-4,
        warmup_steps=30,
        pad_id=tokenizer.pad_id,
        device=device,
    )

    # Generation with KV cache
    prompt = "The quick "
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(
        prompt_ids,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_id,
        temperature=0.8,
        top_k=40,
        do_sample=True,
    )
    text = tokenizer.decode(generated[0].tolist(), skip_special=False)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Generated: {text}")

    greedy = model.generate(
        prompt_ids,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_id,
        do_sample=False,
    )
    print(f"Greedy:   {tokenizer.decode(greedy[0].tolist(), skip_special=False)}")


if __name__ == "__main__":
    main()
