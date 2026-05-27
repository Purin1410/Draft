from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from utils.utils import Hypothesis
from utils.vocab_info import VocabInfo

from .decoder import Decoder, TAMERDecoderOutput
from .encoder import Encoder


class TAMER(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_info: VocabInfo,
    ):
        super().__init__()
        mcfg = config["model"]
        tamer_cfg = mcfg.get("tamer", {})
        struct_head_cfg = tamer_cfg.get("struct_head", {})
        tree_rescore_cfg = tamer_cfg.get("tree_rescore", {})

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

        self.tree_rescore_cfg = tree_rescore_cfg

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
            struct_head_enabled=bool(struct_head_cfg.get("enabled", False)),
            struct_nhead=struct_head_cfg.get("nhead", nhead),
            struct_num_layers=struct_head_cfg.get("num_layers", 1),
            struct_dim_feedforward=struct_head_cfg.get(
                "dim_feedforward", dim_feedforward
            ),
            struct_dropout=struct_head_cfg.get("dropout", dropout),
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> TAMERDecoderOutput:
        feature, mask = self.encoder(img, img_mask)
        feature = torch.cat((feature, feature), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        return self.decoder(feature, mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        current_epoch: int = 0,
        **kwargs,
    ) -> List[Hypothesis]:
        feature, mask = self.encoder(img, img_mask)
        return self.decoder.beam_search(
            [feature],
            [mask],
            beam_size,
            max_len,
            alpha,
            early_stopping,
            temperature,
            tree_rescore_cfg=self.tree_rescore_cfg,
            current_epoch=current_epoch,
        )
