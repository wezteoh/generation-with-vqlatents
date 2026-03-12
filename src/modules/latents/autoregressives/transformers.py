"""
Unconditional prior over VQ codebook indices via causal transformer.
Contains decoder-only GPT (nanoGPT-style) and LatentTransformer (nn.Module).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.modules.autoencoders import VQAutoencoder


# -------- GPT (decoder-only, nanoGPT-style) --------


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (nanoGPT style)."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Single-head or multi-head causal self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_head)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Two-layer MLP with GELU (nanoGPT style)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Decoder block: pre-norm attention + pre-norm MLP with residuals."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalGPT(nn.Module):
    """
    Decoder-only GPT for autoregressive next-token prediction.
    Mirrors nanoGPT: token + position embeddings, stacked Blocks,
    final ln_f and lm_head (no weight tying).
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(self.config) for _ in range(n_layer)]),
                ln_f=LayerNorm(n_embd, bias=bias),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_block_size(self) -> int:
        return self.config.block_size

    def forward(
        self,
        indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Any]:
        device = indices.device
        B, T = indices.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer["wte"](indices)
        pos_emb = self.transformer["wpe"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        return logits, None


# -------- Latent prior (VQ + GPT, no Lightning) --------


def _disabled_train(self: nn.Module, mode: bool = True) -> "nn.Module":
    """Disable train/eval toggling so the module stays in eval (e.g. frozen VQ)."""
    return self


class IdentityPermuter(nn.Module):
    """No-op permuter; indices pass through as-is (forward and reverse)."""

    def forward(
        self,
        indices: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        return indices


class SOSProvider(nn.Module):
    """Provides a fixed SOS token of shape (batch_size, 1) for unconditional prior."""

    def __init__(self, sos_token: int = 0):
        super().__init__()
        self.sos_token = sos_token

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, 1),
            self.sos_token,
            dtype=torch.long,
            device=device,
        )


class LatentTransformer(nn.Module):
    """
    Unconditional autoregressive prior over flattened VQ codebook indices.
    Pure nn.Module; training_step/validation_step/configure_optimizers live in the interface.
    """

    def __init__(
        self,
        first_stage_model: VQAutoencoder,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        transformer_dropout: float = 0.0,
        first_stage_key: str = "image",
        pkeep: float = 1.0,
        sos_token: int = 0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Optional[list] = None,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.pkeep = pkeep
        self.sos_token = sos_token

        self.first_stage_model = first_stage_model
        self.first_stage_model.eval()
        self.first_stage_model.train = _disabled_train.__get__(self.first_stage_model)

        self.permuter = IdentityPermuter()
        self.sos_provider = SOSProvider(sos_token)
        self.transformer = CausalGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=transformer_dropout,
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys or [])

    def init_from_ckpt(self, path: str, ignore_keys: Optional[list] = None) -> None:
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location="cpu")
        state_dict = sd["state_dict"] if "state_dict" in sd else sd
        for k in list(state_dict.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    del state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        print(f"Prior restored from {path}")

    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2]
        b = quant_z.shape[0]
        if indices.dim() == 1:
            indices = rearrange(indices, "(b l) -> b l", b=b)
        elif indices.dim() > 2:
            indices = rearrange(indices, "b ... -> b (...)", b=b)
        indices = self.permuter(indices, reverse=False)
        return quant_z, indices

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, z_indices = self.encode_to_z(x)
        c_indices = self.sos_provider(z_indices.shape[0], z_indices.device)

        if self.training and self.pkeep < 1.0:
            mask = (
                torch.bernoulli(
                    self.pkeep * torch.ones(z_indices.shape, device=z_indices.device)
                )
                .round()
                .to(dtype=torch.int64)
            )
            r_indices = torch.randint_like(
                z_indices, self.transformer.config.vocab_size, device=z_indices.device
            )
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        target = z_indices
        logits, _ = self.transformer(cz_indices[:, :-1])
        logits = logits[:, c_indices.shape[1] - 1:]
        return logits, target

    def _top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if k is None or k <= 0:
            return logits
        # Clamp k to the available vocabulary / logits size to avoid RuntimeError
        k = min(k, logits.size(-1))
        v, _ = torch.topk(logits, k, dim=-1)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        steps: int,
        temperature: float = 1.0,
        sample: bool = True,
        top_k: Optional[int] = None,
        callback: Optional[Any] = None,
    ) -> torch.Tensor:
        """Autoregressively sample latent indices only. Shape (batch_size, steps)."""
        callback = callback or (lambda k: None)
        device = next(self.transformer.parameters()).device
        c = self.sos_provider(batch_size, device)
        x = c
        block_size = self.transformer.get_block_size()
        self.transformer.eval()
        for k in range(steps):
            callback(k)
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits, _ = self.transformer(x_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            x = torch.cat((x, ix), dim=1)
        indices = x[:, c.shape[1]:]
        return indices

    @torch.no_grad()
    def sample_natural(
        self,
        batch_size: int,
        z_shape: tuple[int, ...],
        temperature: float = 1.0,
        sample: bool = True,
        top_k: Optional[int] = None,
        callback: Optional[Any] = None,
    ) -> torch.Tensor:
        """Sample latent indices then decode to images. Returns (B, C, H, W)."""
        steps = z_shape[2] * z_shape[3]
        indices = self.sample_latents(
            batch_size=batch_size,
            steps=steps,
            temperature=temperature,
            sample=sample,
            top_k=top_k,
            callback=callback,
        )
        return self.decode_to_img(indices, z_shape)

    @torch.no_grad()
    def decode_to_img(
        self,
        index: torch.Tensor,
        zshape: tuple[int, ...],
    ) -> torch.Tensor:
        """Decode flat indices (B, L) to images using frozen first-stage decoder."""
        index = self.permuter(index, reverse=True)
        embed_dim = self.first_stage_model.quantize.e_dim
        bhwc = (zshape[0], zshape[2], zshape[3], embed_dim)
        flat_index = rearrange(index, "b l -> (b l)")
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            flat_index, shape=bhwc
        )
        return self.first_stage_model.decode(quant_z)

    def get_input(self, key: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch[key]
        if x.dim() == 3:
            x = x[..., None]
        if x.dim() == 4:
            if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
                x = rearrange(x, "b h w c -> b c h w").contiguous()
            else:
                x = x.contiguous()
        if x.dtype == torch.double:
            x = x.float()
        return x
