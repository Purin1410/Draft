from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from utils.utils import Hypothesis
from utils.vocab_info import VocabInfo

from .decoder import Decoder
from .encoder import Encoder
from .position_decoder import PosDecoder


def _resolve_coverage_mask_token_ids(config: Dict[str, Any], vocab_info: VocabInfo) -> Optional[List[int]]:
    mcfg = config["model"]
    correction_cfg = mcfg.get("implicit_attention_correction", {})
    if not correction_cfg.get("enabled", True):
        return None

    words = getattr(vocab_info, "words", None)
    word2idx = getattr(words, "word2idx", None)
    if word2idx is None:
        raise ValueError("vocab_info.words.word2idx is required for implicit attention correction.")

    mask_tokens = correction_cfg.get("mask_tokens", ["{", "^", "_"])
    missing = [token for token in mask_tokens if token not in word2idx]
    if missing:
        raise KeyError(f"Missing implicit attention correction mask tokens: {missing}")
    return [int(word2idx[token]) for token in mask_tokens]


class PosFormer(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_info: VocabInfo,
    ):
        super().__init__()
        mcfg = config["model"]

        d_model = mcfg.get("d_model", 256)
        growth_rate = mcfg.get("growth_rate")
        num_layers = mcfg.get("num_layers", 16)
        reduction = mcfg.get("reduction", 0.5)
        bottleneck = mcfg.get("bottleneck", True)
        use_dropout = mcfg.get("use_dropout", True)
        densenet_dropout = mcfg.get("encoder_dropout", 0.2)

        nhead = mcfg.get("nhead", 8)
        num_decoder_layers = mcfg.get("num_decoder_layers", 3)
        dim_feedforward = mcfg.get("dim_feedforward", 1024)
        dropout = mcfg.get("decoder_dropout", 0.3)
        dc = mcfg.get("dc", 32)
        cross_coverage = mcfg.get("cross_coverage", True)
        self_coverage = mcfg.get("self_coverage", True)
        position_cfg = mcfg.get("position", {})

        self.position_enabled = bool(position_cfg.get("enabled", True))
        coverage_mask_token_ids = _resolve_coverage_mask_token_ids(config, vocab_info)

        self.encoder = Encoder(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            reduction=reduction,
            bottleneck=bottleneck,
            use_dropout=use_dropout,
            densenet_dropout=densenet_dropout,
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_info=vocab_info,
            coverage_mask_token_ids=coverage_mask_token_ids,
        )
        self.pos_decoder = (
            PosDecoder(
                config=config,
                vocab_info=vocab_info,
                coverage_mask_token_ids=coverage_mask_token_ids,
            )
            if self.position_enabled
            else None
        )

    def forward(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        tgt: LongTensor,
        pos_tgt: Optional[FloatTensor] = None,
        return_aux: bool = False, capture_embed: bool = False,
        capture_cross_attn: bool = False, capture_self_attn: bool = False
    ):
        feature, mask = self.encoder(img, img_mask)
        feature = torch.cat((feature, feature), dim=0)
        mask = torch.cat((mask, mask), dim=0)

        decoder_out = self.decoder(
            feature,
            mask,
            tgt,
            return_aux=return_aux,
            capture_embed=capture_embed,
            capture_cross_attn=capture_cross_attn,
            capture_self_attn=capture_self_attn,
        )
        if return_aux:
            word_logits, aux = decoder_out
        else:
            word_logits = decoder_out
            aux = None

        if not self.position_enabled:
            layer_logits, pos_logits = None, None
        else:
            if pos_tgt is None:
                raise ValueError("pos_tgt is required when PosFormer position branch is enabled.")
            layer_logits, pos_logits = self.pos_decoder(feature, mask, tgt, pos_tgt)
            
        if return_aux:
            return (word_logits, layer_logits, pos_logits), aux
        return word_logits, layer_logits, pos_logits

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        feature, mask = self.encoder(img, img_mask)
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature, return_nbest=kwargs.get("return_nbest", False)
        )
