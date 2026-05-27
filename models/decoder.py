from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from utils.vocab_info import VocabInfo

from .pos_enc import WordPosEnc
from .transformer.arm import AttentionRefinementModule
from .transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from utils.generation_utils import DecodeModel


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_info: VocabInfo,
        sccm: Optional[Dict[str, Any]] = None,
        fusion: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.vocab_info = vocab_info

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_info.vocab_size, d_model), nn.LayerNorm(d_model)
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
        )

        sccm = sccm or {}
        fusion = fusion or {}
        if fusion.get("type", "gated") != "gated":
            raise ValueError(f"Unsupported fusion type: {fusion.get('type')}")

        self.SCCM = SCCM(
            d_model=d_model,
            nhead=sccm.get("nhead", 8),
            dim_feedforward=sccm.get("dim_feedforward", 1024),
            dropout=sccm.get("dropout", 0.3),
            num_layers=sccm.get("num_layers", 1),
        )
        self.fusion = FusionModule(d_model)
        self.exp_proj = nn.Linear(d_model, vocab_info.vocab_size)
        self.imp_proj = nn.Sequential(nn.ReLU(), nn.Linear(d_model, vocab_info.vocab_size))
        self.fusion_proj = nn.Sequential(
            nn.ReLU(inplace=True), nn.Linear(d_model, vocab_info.vocab_size)
        )

        self._causal_mask_cache = {}

    def _build_attention_mask(self, length, device=None, dtype=torch.bool):
        if device is None:
            device = self.device
        cache_key = (device.type, device.index, str(dtype))
        cached = self._causal_mask_cache.get(cache_key)
        if cached is not None and cached.size(0) >= length:
            return cached[:length, :length]

        # lazily create causal attention mask (upper triangular = True)
        mask = torch.full(
            (length, length), fill_value=1, dtype=dtype, device=device
        )
        mask.triu_(1)  # zero out the lower diagonal
        self._causal_mask_cache[cache_key] = mask
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        Tuple[FloatTensor, FloatTensor, FloatTensor]
            explicit, implicit, and fusion logits, each [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l, device=tgt.device)
        tgt_pad_mask = tgt == self.vocab_info.pad_id
        
        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        exp_hidden = rearrange(out, "l b d -> b l d")
        imp_hidden = self.SCCM(exp_hidden, tgt_mask, tgt_pad_mask)
        fusion_hidden = self.fusion(exp_hidden, imp_hidden)

        exp_out = self.exp_proj(exp_hidden)
        imp_out = self.imp_proj(imp_hidden)
        fusion_out = self.fusion_proj(fusion_hidden)

        return exp_out, imp_out, fusion_out


    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        _, _, fusion_out = self(src[0], src_mask[0], input_ids)
        return fusion_out


class SCCM(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        num_layers: int = 1,
    ):
        super().__init__()
        self.te = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )

    def forward(
        self,
        out: FloatTensor,
        tgt_mask: LongTensor,
        src_key_padding_mask: LongTensor,
    ) -> FloatTensor:
        out = rearrange(out, "b t d -> t b d")
        out = self.te(
            src=out,
            mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return rearrange(out, "t b d -> b t d")


class FusionModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w_att = nn.Linear(2 * d_model, d_model)

    def forward(self, e_feature: FloatTensor, i_feature: FloatTensor) -> FloatTensor:
        feature = torch.cat((e_feature, i_feature), dim=2)
        gate = torch.sigmoid(self.w_att(feature))
        return gate * i_feature + (1 - gate) * e_feature
