from typing import List, Dict, Any

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


from datamodule.vocab import VocabInfo

class CoMER(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_info: VocabInfo,
    ):
        super().__init__()
        mcfg = config["model"]
        # encoder
        d_model             = mcfg.get("d_model", 256)
        growth_rate         = mcfg.get("growth_rate")
        num_layers          = mcfg.get("num_layers", 16)
        reduction           = mcfg.get("reduction", 0.5)
        bottleneck          = mcfg.get("bottleneck", True)
        use_dropout         = mcfg.get("use_dropout", True)
        densenet_dropout    = mcfg.get("encoder_dropout", 0.2)
        # decoder
        nhead               = mcfg.get("nhead", 8)
        num_decoder_layers  = mcfg.get("num_decoder_layers", 3)
        dim_feedforward     = mcfg.get("dim_feedforward", 1024)
        dropout             = mcfg.get("decoder_dropout", 0.3)
        dc                  = mcfg.get("dc", 32)
        cross_coverage      = mcfg.get("cross_coverage", True)
        self_coverage       = mcfg.get("self_coverage", True)

        use_tree_bias       = mcfg.get("use_tree_bias", True)
        tree_bias_num_buckets = mcfg.get("tree_bias_num_buckets", 16)
        tree_bias_mode      = mcfg.get("tree_bias_mode", "full")
        tree_bias_layers    = mcfg.get("tree_bias_layers", "all")
        tree_bias_rel_set   = mcfg.get("tree_bias_rel_set", "full")

        self.encoder = Encoder(
            d_model=d_model, 
            growth_rate=growth_rate, 
            num_layers=num_layers,
            reduction=reduction,
            bottleneck=bottleneck,
            use_dropout=use_dropout,
            densenet_dropout=densenet_dropout
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
            use_tree_bias=use_tree_bias,
            tree_bias_num_buckets=tree_bias_num_buckets,
            tree_bias_mode=tree_bias_mode,
            tree_bias_layers=tree_bias_layers,
            tree_bias_rel_set=tree_bias_rel_set,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, rel_ids: torch.LongTensor = None
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        out = self.decoder(feature, mask, tgt, rel_ids=rel_ids)

        return out

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
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
