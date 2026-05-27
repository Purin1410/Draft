from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from utils.vocab_info import VocabInfo

from .decoder import _build_transformer_decoder
from .pos_enc import WordPosEnc


class PosDecoder(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_info: VocabInfo,
        coverage_mask_token_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        mcfg = config["model"]
        position_cfg = mcfg.get("position", {})

        d_model = int(mcfg.get("d_model", 256))
        nhead = int(mcfg.get("nhead", 8))
        num_decoder_layers = int(mcfg.get("num_decoder_layers", 3))
        dim_feedforward = int(mcfg.get("dim_feedforward", 1024))
        dropout = float(position_cfg.get("dropout", mcfg.get("decoder_dropout", 0.3)))
        dc = int(mcfg.get("dc", 32))
        cross_coverage = bool(mcfg.get("cross_coverage", True))
        self_coverage = bool(mcfg.get("self_coverage", True))

        self.vocab_info = vocab_info
        self.label_depth = int(position_cfg.get("label_depth", 5))
        self.num_layer_classes = int(position_cfg.get("num_layer_classes", 5))
        self.num_pos_classes = int(position_cfg.get("num_pos_classes", 6))
        mlp_hidden_dim = int(position_cfg.get("mlp_hidden_dim", 128))
        embed_hidden_dim = position_cfg.get("embed_hidden_dim", None)

        if embed_hidden_dim is None:
            self.pos_embed = nn.Sequential(
                nn.Linear(self.label_depth, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            embed_hidden_dim = int(embed_hidden_dim)
            self.pos_embed = nn.Sequential(
                nn.Linear(self.label_depth, embed_hidden_dim),
                nn.GELU(),
                nn.Linear(embed_hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.pos_enc = WordPosEnc(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            coverage_mask_token_ids=coverage_mask_token_ids,
        )
        self.layernum_proj = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.num_layer_classes),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.num_pos_classes),
        )
        self._causal_mask_cache = {}

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_attention_mask(self, length, device=None, dtype=torch.bool):
        if device is None:
            device = self.device
        cache_key = (device.type, device.index, str(dtype))
        cached = self._causal_mask_cache.get(cache_key)
        if cached is not None and cached.size(0) >= length:
            return cached[:length, :length]

        mask = torch.full((length, length), fill_value=1, dtype=dtype, device=device)
        mask.triu_(1)
        self._causal_mask_cache[cache_key] = mask
        return mask

    def forward(
        self,
        src: FloatTensor,
        src_mask: LongTensor,
        tgt: LongTensor,
        pos_tgt: FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        _, length = tgt.size()
        tgt_mask = self._build_attention_mask(length)
        tgt_pad_mask = tgt == self.vocab_info.pad_id
        tgt_vocab = tgt

        pos_tgt = self.pos_embed(pos_tgt.to(dtype=src.dtype))
        pos_tgt = self.pos_enc(pos_tgt)
        pos_tgt = self.norm(pos_tgt)

        height = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        pos_tgt = rearrange(pos_tgt, "b l d -> l b d")

        out = self.model(
            tgt=pos_tgt,
            memory=src,
            height=height,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
        out = rearrange(out, "l b d -> b l d")
        return self.layernum_proj(out), self.pos_proj(out)
