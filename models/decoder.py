from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass
class TAMERDecoderOutput:
    logits: FloatTensor
    struct_logits: Optional[FloatTensor] = None


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
        struct_head_enabled: bool = False,
        struct_nhead: int = 8,
        struct_num_layers: int = 1,
        struct_dim_feedforward: int = 1024,
        struct_dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_info = vocab_info
        self.struct_head_enabled = struct_head_enabled

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

        self.proj = nn.Linear(d_model, vocab_info.vocab_size)
        self.struct_sim = (
            StructSim(
                d_model=d_model,
                nhead=struct_nhead,
                num_layers=struct_num_layers,
                dim_feedforward=struct_dim_feedforward,
                dropout=struct_dropout,
            )
            if struct_head_enabled
            else None
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
    ) -> TAMERDecoderOutput:
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
        FloatTensor
            [b, l, vocab_size]
        """
        B_tgt, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == self.vocab_info.pad_id
        
        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        hidden = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        struct_logits = (
            self.struct_sim(hidden, tgt_pad_mask)
            if self.struct_sim is not None
            else None
        )
        out = rearrange(hidden, "l b d -> b l d")
        logits = self.proj(out)

        return TAMERDecoderOutput(logits=logits, struct_logits=struct_logits)


    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        return self(src[0], src_mask[0], input_ids).logits

    def transform_with_struct(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> TAMERDecoderOutput:
        assert len(src) == 1 and len(src_mask) == 1
        return self(src[0], src_mask[0], input_ids)


class StructSimOneDir(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.trm = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_sim = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(d_model, 1))

    def forward(self, tgt: FloatTensor, tgt_key_padding_mask: LongTensor) -> FloatTensor:
        tgt = self.trm(src=tgt, src_key_padding_mask=tgt_key_padding_mask)
        q = rearrange(self.to_q(tgt), "t b d -> b t () d")
        k = rearrange(self.to_k(tgt), "l b d -> b () l d")
        sim = self.to_sim(q + k).squeeze(-1)
        return sim.masked_fill(tgt_key_padding_mask[:, None, :], float("-inf"))


class StructSim(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.l2r_struct_sim = StructSimOneDir(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.r2l_struct_sim = StructSimOneDir(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, out: FloatTensor, tgt_key_padding_mask: LongTensor) -> FloatTensor:
        l2r_out, r2l_out = torch.chunk(out, 2, dim=1)
        l2r_mask, r2l_mask = torch.chunk(tgt_key_padding_mask, 2, dim=0)
        l2r_sim = self.l2r_struct_sim(l2r_out, l2r_mask)
        r2l_sim = self.r2l_struct_sim(r2l_out, r2l_mask)
        return torch.cat((l2r_sim, r2l_sim), dim=0)

