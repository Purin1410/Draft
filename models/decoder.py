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
from .transformer.tree_bias import TreeRelationBuilder, TreeRelativeBias
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
    tree_bias_layers: str = "all",
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

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm, tree_bias_layers=tree_bias_layers)
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
        use_tree_bias: bool = True,
        tree_bias_num_buckets: int = 16,
        tree_bias_mode: str = "full",
        tree_bias_layers: str = "all",
        tree_bias_rel_set: str = "full",
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
            tree_bias_layers=tree_bias_layers,
        )

        self.proj = nn.Linear(d_model, vocab_info.vocab_size)

        # -----------------------------
        # Tree-structure relative bias
        # -----------------------------
        self.use_tree_bias = bool(use_tree_bias)
        self.tree_bias_layers = tree_bias_layers

        if self.use_tree_bias:
            if vocab_info is None or vocab_info.words is None or not hasattr(vocab_info.words, "idx2word"):
                raise ValueError("Tree bias requires vocab_info.words.idx2word")

            self._tree_builder = TreeRelationBuilder(
                id2tok=vocab_info.words.idx2word,
                pad_id=vocab_info.pad_id,
                num_buckets=tree_bias_num_buckets,
                mode=tree_bias_mode,
                rel_set=tree_bias_rel_set,
            )
            self._tree_rel_bias = TreeRelativeBias(
                num_heads=nhead,
                num_relations=self._tree_builder.num_relations,
            )
        else:
            self._tree_builder = None
            self._tree_rel_bias = None
        # Causal mask cache: keyed by (device_type, device_index, dtype_str)
        # so CPU->CUDA or dtype changes don't reuse a stale/wrong-device mask.
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
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor, rel_ids: Optional[LongTensor] = None
    ) -> FloatTensor:
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

        rel_bias = None
        if self.use_tree_bias and self._tree_rel_bias is not None:
            if rel_ids is None:
                rel_ids = self._tree_builder.build(tgt)
            rel_bias = self._tree_rel_bias(rel_ids, flatten=True)

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
            rel_bias=rel_bias,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out


    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        return self(src[0], src_mask[0], input_ids)

