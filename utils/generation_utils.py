from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Hypothesis, ce_loss, to_tgt_output
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor
from utils.vocab_info import VocabInfo
from .beam_search import BeamSearchScorer
from .analysis_logging import BeamCandidate, BeamSearchOutput


# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


def _strip_generated_boundaries_cpu(
    seq: torch.Tensor,
    sos_id: int,
    eos_id: int,
) -> torch.Tensor:
    """Strip boundary tokens (leading start token, trailing terminal token)
    from a 1-D generated sequence tensor.

    Rules
    -----
    - Remove the first token if it equals sos_id or eos_id (start token).
    - Remove the last token if it equals eos_id or sos_id (terminal token).
    - Interior tokens are never removed (even if they happen to be sos/eos).
    - Returns empty tensor safely.

    Parameters
    ----------
    seq : torch.Tensor
        1-D tensor of token ids (no PAD, already filtered). Must be on CPU.
    sos_id : int
    eos_id : int

    Returns
    -------
    torch.Tensor
        Cleaned 1-D tensor comparable to ground-truth label indices.
    """
    assert seq.device.type == "cpu", "Boundary stripping must run on CPU tensors only"
    if seq.numel() == 0:
        return seq

    boundary_ids = {sos_id, eos_id}

    # Remove leading start token
    start = 0
    if seq[0].item() in boundary_ids:
        start = 1

    # Remove trailing terminal token
    end = seq.numel()
    if end > start and seq[end - 1].item() in boundary_ids:
        end -= 1

    return seq[start:end]



class DecodeModel(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """decode one step

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        raise NotImplementedError("This is an abstract method.")


    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        return_nbest: bool = False,
    ) -> Union[List[Hypothesis], List[BeamSearchOutput]]:
        """run beam search to decode

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        beam_size : int
        max_len : int
        alpha : float
        early_stopping : bool
        temperature : float
        return_nbest : bool
            If True, return List[BeamSearchOutput] with all candidates per sample.
            If False (default), return List[Hypothesis] (backward compatible).

        Returns
        -------
        Union[List[Hypothesis], List[BeamSearchOutput]]
        """
        batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction
        batch_beam_size = batch_size * beam_size
        half_bb_size = batch_beam_size // 2
        real_batch = batch_size // 2  # original samples

        for i in range(len(src)):
            # Bidirectional beam search: duplicate encoder features for l2r + r2l directions.
            # This copy is done ONCE here, before the decode loop, not inside it.
            # TODO: if memory is very tight, keep src as [B,...] and use batch-index
            #       indirection inside the loop instead of materialising the copy.
            src[i] = torch.cat((src[i], src[i]), dim=0)
            src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

        l2r = torch.full(
            (batch_size // 2, 1),
            fill_value=self.vocab_info.sos_id,
            dtype=torch.long,
            device=self.device,
        )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=self.vocab_info.eos_id,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)

        beam_scorer = BeamSearchScorer(
            batch_size, beam_size, alpha, early_stopping, self.device, self.vocab_info
        )

        # first beam search
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )

        # reverse half last
        for i in range(half_bb_size, batch_beam_size):
            hyps[i] = torch.flip(hyps[i], dims=[0])

        lens = [len(h) + 1 for h in hyps]  # plus to append start token
        r2l_tgt, r2l_out = to_tgt_output(
            hyps[:half_bb_size], "r2l", self.device, self.vocab_info.sos_id, self.vocab_info.eos_id, self.vocab_info.pad_id, pad_to_len=max(lens)
        )
        l2r_tgt, l2r_out = to_tgt_output(
            hyps[half_bb_size:], "l2r", self.device, self.vocab_info.sos_id, self.vocab_info.eos_id, self.vocab_info.pad_id, pad_to_len=max(lens)
        )
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        # calculate final score
        rev_scores = self._rate(src, src_mask, tgt, out, alpha, temperature)
        rev_scores = torch.cat(
            (rev_scores[half_bb_size:], rev_scores[:half_bb_size]), dim=0
        )
        scores = scores + rev_scores

        # [2 * b, beam_size]
        scores_2d = rearrange(scores, "(b m) -> b m", b=batch_size)
        l2r_scores, r2l_scores = torch.chunk(scores_2d, 2, dim=0)
        # [b, 2 * beam_size]
        combined_scores = torch.cat((l2r_scores, r2l_scores), dim=1)
        # [batch_size, ]
        best_scores, best_indices = torch.max(combined_scores, dim=1)
        best_split = best_indices // beam_size
        best_indices_in_beam = best_indices % beam_size
        batch_indices = torch.arange(
            0, real_batch, dtype=torch.long, device=self.device
        )
        best_flat_indices = (
            best_split * half_bb_size + batch_indices * beam_size + best_indices_in_beam
        )

        # Post-decode CPU conversion — .cpu().tolist() is allowed here (outside hot loop)
        best_flat_cpu = best_flat_indices.cpu().tolist()
        best_scores_cpu = best_scores.cpu().tolist()

        ret: List[Hypothesis] = []
        for idx, score in zip(best_flat_cpu, best_scores_cpu):
            hpy = Hypothesis(hyps[idx].cpu(), score, "l2r")
            ret.append(hpy)

        if not return_nbest:
            return ret

        # Build n-best output per sample
        nbest_results: List[BeamSearchOutput] = []
        scores_cpu = scores.cpu()
        for b in range(real_batch):
            sample_candidates = []
            # l2r candidates: indices [b*beam_size .. (b+1)*beam_size) in first half
            for k in range(beam_size):
                l2r_idx = b * beam_size + k
                l2r_score = scores_cpu[l2r_idx].item()
                sample_candidates.append(
                    BeamCandidate(seq=hyps[l2r_idx].cpu(), score=l2r_score, direction="l2r")
                )
            # r2l candidates: indices [half_bb_size + b*beam_size .. half_bb_size + (b+1)*beam_size)
            for k in range(beam_size):
                r2l_idx = half_bb_size + b * beam_size + k
                r2l_score = scores_cpu[r2l_idx].item()
                sample_candidates.append(
                    BeamCandidate(seq=hyps[r2l_idx].cpu(), score=r2l_score, direction="r2l")
                )
            # Sort descending by score
            sample_candidates.sort(key=lambda c: c.score, reverse=True)
            nbest_results.append(BeamSearchOutput(best=ret[b], candidates=sample_candidates))

        return nbest_results

    def _beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        beam_scorer: BeamSearchScorer,
        beam_size: int,
        max_len: int,
        temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        batch_size, cur_len = input_ids.shape
        vocab_size = self.vocab_info.vocab_size

        beam_scores = torch.zeros(batch_size, dtype=torch.float, device=self.device)

        while cur_len < max_len and not beam_scorer.is_done():
            next_token_logits = (
                self.transform(src, src_mask, input_ids)[:, -1, :] / temperature
            )
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            
            reshape_size = next_token_scores.shape[0] // batch_size
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            if cur_len == 1:
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)

            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )

            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
            )
            cur_len += 1

        return beam_scorer.finalize(input_ids, beam_scores)

    def _rate(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        tgt: LongTensor,
        out: LongTensor,
        alpha: float,
        temperature: float,
    ) -> FloatTensor:
        """rate tgt and output

        Parameters
        ----------
        src : List[FloatTensor]
            [b * beam_size, t, d]
        src_mask : List[LongTensor]
            [b * beam_size, t]
        tgt : LongTensor
            [b * beam_size, l]
        out : LongTensor
            [b * beam_size, l]
        alpha : float
        temperature : float

        Returns
        -------
        FloatTensor
            [b * beam_size]
        """
        b = tgt.shape[0]
        out_hat = self.transform(src, src_mask, tgt) / temperature
        loss = ce_loss(out_hat, out, ignore_idx=self.vocab_info.pad_id, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)
        mask = tgt == self.vocab_info.pad_id
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty
        return loss
