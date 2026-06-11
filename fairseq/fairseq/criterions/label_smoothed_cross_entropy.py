# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    noise_lam: float = field(
        default=1.0,
        metadata={"help": "weight for router (noise classifier) loss: total_loss = asr_loss + noise_lam * noise_loss"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        asr_loss = loss  # noise_loss 더하기 전 ASR loss

        # speaker_load / spk_label_cnt: noise_label만 있으면 moe_mode=none에서도 집계
        if "noise_label" in sample:
            noise_label = sample["noise_label"]
            spk_cnts = {}
            for n in range(4):
                mask_n = noise_label == n
                if mask_n.any():
                    spk_cnts[n] = mask_n.sum().item()
        else:
            noise_label = None
            spk_cnts = {}

        if "noise_label" in sample and "noise_logits" in net_output[1]:
            noise_logits = net_output[1]["noise_logits"]                   # [T, B, 4]
            noise_label = noise_label.to(noise_logits.device)              # [B]
            T, B, E = noise_logits.shape
            log_probs = torch.log(noise_logits.clamp(min=1e-8))            # [T, B, 4]
            frame_nll = -log_probs[:, torch.arange(B, device=noise_logits.device), noise_label]  # [T, B]
            padding_mask = sample["net_input"].get("padding_mask", None)
            if padding_mask is not None:
                valid = ~padding_mask.T                                     # [T, B], True=유효 프레임
                noise_loss = frame_nll[valid].mean()
            else:
                valid = None
                noise_loss = frame_nll.mean()
            loss = asr_loss + noise_loss

            # --- MoE routing 진단 (gradient 불필요) ---
            with torch.no_grad():
                # 1. Routing entropy
                ent = -(noise_logits * noise_logits.clamp(min=1e-8).log()).sum(-1)  # [T, B]
                if valid is not None:
                    moe_entropy_sum = ent[valid].sum().item()
                    moe_token_count = valid.sum().item()
                else:
                    moe_entropy_sum = ent.sum().item()
                    moe_token_count = T * B

                # 2. Expert load (전체 토큰 합산)
                expert_sums = noise_logits.sum(0).sum(0)  # [E]

                # 3. 라우터 예측 화자 수 기댓값 (발화 단위, num_speaker_avg와 동일 단위)
                per_utt = noise_logits.mean(0)  # [B, E]
                spk_values = torch.arange(E, device=noise_logits.device, dtype=noise_logits.dtype)
                router_pred_spk_sum = (per_utt * spk_values).sum(-1).sum().item()  # [B] → scalar
        else:
            noise_loss = None
            moe_entropy_sum = moe_token_count = 0
            expert_sums = None
            router_pred_spk_sum = 0

        # speaker_load: noise_logits 유무와 무관하게 noise_label 분포를 항상 집계
        spk_cnts = {}
        if noise_label is not None:
            for n in range(4):
                mask_n = noise_label == n
                if mask_n.any():
                    spk_cnts[n] = mask_n.sum().item()

        logging_output = {
            "loss": loss.data,                                          # total loss (asr + noise)
            "asr_loss": asr_loss.data,                                  # ASR label-smoothed loss
            "nll_loss": nll_loss.data,                                  # ASR NLL loss (ppl용)
            "noise_loss": noise_loss.data if noise_loss is not None else 0,  # noise classifier loss
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            # MoE routing 진단
            "moe_entropy_sum": moe_entropy_sum,
            "moe_token_count": moe_token_count,
            "router_pred_spk_sum": router_pred_spk_sum,
        }
        if expert_sums is not None:
            for k in range(E):
                logging_output[f"moe_e{k}_sum"] = expert_sums[k].item()
        for n, cnt in spk_cnts.items():
            logging_output[f"spk{n}_cnt"] = cnt
        if noise_label is not None:
            logging_output["spk_label_sum"] = noise_label.float().sum().item()
            logging_output["spk_label_cnt"] = noise_label.numel()
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        asr_loss_sum = sum(log.get("asr_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        noise_loss_sum = sum(log.get("noise_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "asr_loss", asr_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "noise_loss", noise_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        def _sum(key):
            return sum(log.get(key, 0) for log in logging_outputs)

        # --- MoE routing 진단 (5개) ---
        token_count = _sum("moe_token_count")
        total_utts  = _sum("spk_label_cnt")

        # 1. Routing entropy (scalar, wandb graph)
        if token_count > 0:
            metrics.log_scalar(
                "moe_routing_entropy", _sum("moe_entropy_sum") / token_count, round=4
            )

        # 2. moe_expert_load=[e0,e1,e2,e3] — moe_mode != none일 때만 출력
        if token_count > 0:
            for k in range(4):
                metrics.log_scalar(f"_moe_e{k}", _sum(f"moe_e{k}_sum") / token_count, round=4)
            metrics.log_derived(
                "moe_expert_load",
                lambda m: "[{:.4f},{:.4f},{:.4f},{:.4f}]".format(
                    *[m[f"_moe_e{k}"].smoothed_value if f"_moe_e{k}" in m else 0.0 for k in range(4)]
                )
            )

        # 3. speaker_load=[s0,s1,s2,s3]
        for n in range(4):
            val = _sum(f"spk{n}_cnt") / total_utts if total_utts > 0 else 0.0
            metrics.log_scalar(f"_spk{n}", val, round=4)
        metrics.log_derived(
            "speaker_load",
            lambda m: "[{:.4f},{:.4f},{:.4f},{:.4f}]".format(
                *[m[f"_spk{n}"].smoothed_value if f"_spk{n}" in m else 0.0 for n in range(4)]
            )
        )

        # 4 & 5. 실제 화자 수 평균 vs 라우터 예측 화자 수 평균 (scalar, wandb graph)
        if total_utts > 0:
            metrics.log_scalar(
                "num_speaker_avg", _sum("spk_label_sum") / total_utts, round=3
            )
            metrics.log_scalar(
                "router_pred_spk_avg", _sum("router_pred_spk_sum") / total_utts, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


# #원본 확인

# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import math
# from dataclasses import dataclass, field

# import torch
# from fairseq import metrics, utils
# from fairseq.criterions import FairseqCriterion, register_criterion
# from fairseq.dataclass import FairseqDataclass
# from omegaconf import II


# @dataclass
# class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
#     label_smoothing: float = field(
#         default=0.0,
#         metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
#     )
#     report_accuracy: bool = field(
#         default=False,
#         metadata={"help": "report accuracy metric"},
#     )
#     ignore_prefix_size: int = field(
#         default=0,
#         metadata={"help": "Ignore first N tokens"},
#     )
#     sentence_avg: bool = II("optimization.sentence_avg")


# def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
#     if target.dim() == lprobs.dim() - 1:
#         target = target.unsqueeze(-1)
#     nll_loss = -lprobs.gather(dim=-1, index=target)
#     smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
#     if ignore_index is not None:
#         pad_mask = target.eq(ignore_index)
#         nll_loss.masked_fill_(pad_mask, 0.0)
#         smooth_loss.masked_fill_(pad_mask, 0.0)
#     else:
#         nll_loss = nll_loss.squeeze(-1)
#         smooth_loss = smooth_loss.squeeze(-1)
#     if reduce:
#         nll_loss = nll_loss.sum()
#         smooth_loss = smooth_loss.sum()
#     eps_i = epsilon / (lprobs.size(-1) - 1)
#     loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
#     return loss, nll_loss


# @register_criterion(
#     "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
# )
# class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
#     def __init__(
#         self,
#         task,
#         sentence_avg,
#         label_smoothing,
#         ignore_prefix_size=0,
#         report_accuracy=False,
#     ):
#         super().__init__(task)
#         self.sentence_avg = sentence_avg
#         self.eps = label_smoothing
#         self.ignore_prefix_size = ignore_prefix_size
#         self.report_accuracy = report_accuracy

#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.

#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         net_output = model(**sample["net_input"])
#         loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
#         sample_size = (
#             sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
#         )
#         logging_output = {
#             "loss": loss.data,
#             "nll_loss": nll_loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["target"].size(0),
#             "sample_size": sample_size,
#         }
#         if self.report_accuracy:
#             n_correct, total = self.compute_accuracy(model, net_output, sample)
#             logging_output["n_correct"] = utils.item(n_correct.data)
#             logging_output["total"] = utils.item(total.data)
#         return loss, sample_size, logging_output

#     def get_lprobs_and_target(self, model, net_output, sample):
#         lprobs = model.get_normalized_probs(net_output, log_probs=True)
#         target = model.get_targets(sample, net_output)
#         if self.ignore_prefix_size > 0:
#             if getattr(lprobs, "batch_first", False):
#                 lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
#                 target = target[:, self.ignore_prefix_size :].contiguous()
#             else:
#                 lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
#                 target = target[self.ignore_prefix_size :, :].contiguous()
#         return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

#     def compute_loss(self, model, net_output, sample, reduce=True):
#         lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
#         loss, nll_loss = label_smoothed_nll_loss(
#             lprobs,
#             target,
#             self.eps,
#             ignore_index=self.padding_idx,
#             reduce=reduce,
#         )
#         return loss, nll_loss

#     def compute_accuracy(self, model, net_output, sample):
#         lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
#         mask = target.ne(self.padding_idx)
#         n_correct = torch.sum(
#             lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
#         )
#         total = torch.sum(mask)
#         return n_correct, total

#     @classmethod
#     def reduce_metrics(cls, logging_outputs) -> None:
#         """Aggregate logging outputs from data parallel training."""
#         loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
#         nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
#         ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

#         metrics.log_scalar(
#             "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
#         )
#         metrics.log_scalar(
#             "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
#         )
#         metrics.log_derived(
#             "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
#         )

#         total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
#         if total > 0:
#             metrics.log_scalar("total", total)
#             n_correct = utils.item(
#                 sum(log.get("n_correct", 0) for log in logging_outputs)
#             )
#             metrics.log_scalar("n_correct", n_correct)
#             metrics.log_derived(
#                 "accuracy",
#                 lambda meters: round(
#                     meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
#                 )
#                 if meters["total"].sum > 0
#                 else float("nan"),
#             )

#     @staticmethod
#     def logging_outputs_can_be_summed() -> bool:
#         """
#         Whether the logging outputs returned by `forward` can be summed
#         across workers prior to calling `reduce_metrics`. Setting this
#         to True will improves distributed training speed.
#         """
#         return True
