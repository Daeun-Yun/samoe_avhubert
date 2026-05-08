import math
import math
import os,sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from copy import deepcopy

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from resnet import ResEncoder
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from utils import compute_mask_indices
    from decoder import TransformerDecoder

else:
    from .hubert_pretraining import (
        AVHubertPretrainingConfig,
        AVHubertPretrainingTask,
    )
    from .resnet import ResEncoder
    from .utils import compute_mask_indices
    from .decoder import TransformerDecoder

from omegaconf import II

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})
    moe_mode: str = field(default='none', metadata={'help': "SA-MoE mode: 'none' (standard AV-HuBERT), 'ffn_only' (fc1+fc2), 'full' (q/k/v/out+fc1+fc2)"})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})

class LoRALinear(nn.Module):
    """단일 Linear를 래핑해서 LoRA MoE를 덧붙임. full 모드의 attention projection에 사용."""
    def __init__(self, base_linear, in_features, out_features, noise_logits_ref, num_experts=4, rank=1):
        super().__init__()
        self.base = base_linear
        self.noise_logits_ref = noise_logits_ref
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.ModuleList([
            nn.Linear(rank, out_features, bias=False) for _ in range(num_experts)
        ])
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        for b in self.lora_B:
            nn.init.zeros_(b.weight)

    def forward(self, x):
        noise_logits = self.noise_logits_ref[0]
        a_out = self.lora_A(x)
        expert_outs = torch.stack([b(a_out) for b in self.lora_B], dim=-1)  # [T, B, out_dim, num_experts]
        if noise_logits is not None:
            weights = noise_logits.unsqueeze(2)                              # [T, B, 1, num_experts]
        else:
            T, B = x.size(0), x.size(1)
            weights = x.new_ones(T, B, 1, len(self.lora_B)) / len(self.lora_B)
        return self.base(x) + (expert_outs * weights).sum(dim=-1)


class LoRAMoETransformerLayer(nn.Module):
    """
    TransformerSentenceEncoderLayer 래퍼.
    base_layer의 서브모듈을 직접 self에 등록해서 state_dict key가
    layers.i.fc1.weight 형태를 유지 → pretrain 체크포인트와 key 자동 매칭.
    moe_mode='ffn_only': fc1, fc2에만 LoRA MoE 적용
    moe_mode='full'    : q/k/v/out_proj + fc1 + fc2 모두 적용
    """
    def __init__(self, base_layer, embed_dim, ffn_embed_dim, noise_logits_ref, num_experts=4, rank=1, moe_mode='none'):
        super().__init__()
        # base_layer 서브모듈을 직접 self에 등록 → key에 .base. 안 붙음
        self.self_attn = base_layer.self_attn
        self.self_attn_layer_norm = base_layer.self_attn_layer_norm
        self.fc1 = base_layer.fc1
        self.fc2 = base_layer.fc2
        self.final_layer_norm = base_layer.final_layer_norm
        self.dropout1 = base_layer.dropout1
        self.dropout2 = base_layer.dropout2
        self.dropout3 = base_layer.dropout3
        self.activation_fn = base_layer.activation_fn  # callable, not nn.Module
        self.layer_norm_first = base_layer.layer_norm_first  # bool

        self.noise_logits_ref = noise_logits_ref
        self.moe_mode = moe_mode

        # fc1 LoRA experts: embed_dim → ffn_embed_dim
        self.lora_A_fc1 = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_fc1 = nn.ModuleList([
            nn.Linear(rank, ffn_embed_dim, bias=False) for _ in range(num_experts)
        ])

        # fc2 LoRA experts: ffn_embed_dim → embed_dim
        self.lora_A_fc2 = nn.Linear(ffn_embed_dim, rank, bias=False)
        self.lora_B_fc2 = nn.ModuleList([
            nn.Linear(rank, embed_dim, bias=False) for _ in range(num_experts)
        ])

        nn.init.kaiming_uniform_(self.lora_A_fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_fc2.weight, a=math.sqrt(5))
        for b in self.lora_B_fc1:
            nn.init.zeros_(b.weight)
        for b in self.lora_B_fc2:
            nn.init.zeros_(b.weight)

        # full 모드: attention projection도 LoRALinear로 교체
        if moe_mode == 'full':
            sa = self.self_attn
            sa.q_proj = LoRALinear(sa.q_proj, embed_dim, embed_dim, noise_logits_ref, num_experts, rank)
            sa.k_proj = LoRALinear(sa.k_proj, embed_dim, embed_dim, noise_logits_ref, num_experts, rank)
            sa.v_proj = LoRALinear(sa.v_proj, embed_dim, embed_dim, noise_logits_ref, num_experts, rank)
            sa.out_proj = LoRALinear(sa.out_proj, embed_dim, embed_dim, noise_logits_ref, num_experts, rank)

    def _apply_lora_moe(self, x, lora_A, lora_B):
        # x: [T, B, in_dim]
        noise_logits = self.noise_logits_ref[0]
        a_out = lora_A(x)                                               # [T, B, rank]
        expert_outs = torch.stack([b(a_out) for b in lora_B], dim=-1)  # [T, B, out_dim, num_experts]
        if noise_logits is not None:
            weights = noise_logits.unsqueeze(2)                         # [T, B, 1, num_experts]
        else:
            T, B = x.size(0), x.size(1)
            weights = x.new_ones(T, B, 1, len(lora_B)) / len(lora_B)
        return (expert_outs * weights).sum(dim=-1)                      # [T, B, out_dim]

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None,
                need_weights=False, att_args=None):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(query=x, key=x, value=x,
                                     key_padding_mask=self_attn_padding_mask,
                                     attn_mask=self_attn_mask)
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            fc1_in = x
            x = self.activation_fn(self.fc1(fc1_in) + self._apply_lora_moe(fc1_in, self.lora_A_fc1, self.lora_B_fc1))
            x = self.dropout2(x)
            fc2_in = x
            x = self.fc2(fc2_in) + self._apply_lora_moe(fc2_in, self.lora_A_fc2, self.lora_B_fc2)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(query=x, key=x, value=x,
                                     key_padding_mask=self_attn_padding_mask)
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            fc1_in = x
            x = self.activation_fn(self.fc1(fc1_in) + self._apply_lora_moe(fc1_in, self.lora_A_fc1, self.lora_B_fc1))
            x = self.dropout2(x)
            fc2_in = x
            x = self.fc2(fc2_in) + self._apply_lora_moe(fc2_in, self.lora_A_fc2, self.lora_B_fc2)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg) if cfg.encoder_layers > 0 else None

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        if self.encoder is not None:
            x = self.encoder(x)[0].transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        return x

@register_model("av_hubert", dataclass=AVHubertConfig)
class AVHubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: AVHubertConfig,
        task_cfg: AVHubertPretrainingConfig,
        dictionaries: List[Dictionary],
        **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)


        #de : (1) SA-MOE ROUTER 및 classifier 정의 (moe_mode != 'none' 일 때만)
        self._layer1_output = None
        self._noise_logits_ref = [None]

        if cfg.moe_mode not in ('none', 'None', None):
            self.noise_classifier = nn.Linear(cfg.encoder_embed_dim, 4)

            def _hook(module, input, output):
                self._layer1_output = output[0]  # [T, B, C]
                self._noise_logits_ref[0] = F.softmax(self.noise_classifier(output[0].detach()), dim=-1)  # [T, B, 4]

            self.encoder.layers[1].register_forward_hook(_hook)

            # layers[2]~layers[N-1] → LoRAMoETransformerLayer 로 교체
            num_layers = len(self.encoder.layers)
            for i in range(2, num_layers):
                self.encoder.layers[i] = LoRAMoETransformerLayer(
                    base_layer=self.encoder.layers[i],
                    embed_dim=cfg.encoder_embed_dim,
                    ffn_embed_dim=cfg.encoder_ffn_embed_dim,
                    noise_logits_ref=self._noise_logits_ref,
                    moe_mode=cfg.moe_mode,
                )



        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: AVHubertConfig, task: AVHubertPretrainingTask):
        """Build a new model instance."""

        kwargs = {}
        model = AVHubertModel(cfg, task.cfg, task.dictionaries, **kwargs)
        return model

    def apply_input_mask(self, x, padding_mask, target_list):
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:

            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end-start
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
            logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        #dataset불러오기
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None
        #feature extract : 오디오는 ffn 비디오는 renet 거치기 
        features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
        features_video = self.forward_features(src_video, modality='video')
        modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        if self.training:
            if modality_drop_prob < self.modality_dropout:
                if audio_drop_prob < self.audio_dropout:
                    features_audio = 0 * features_audio
                else:
                    features_video = 0 * features_video
        # 모달리티 퓨젼 (우리는 concat해)
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

        features_pen = features.float().pow(2).mean()
        #기타 전처리 
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.masking_type == 'feature' and mask:
            x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        # 위에서 encoder에다가 transformerencoder를 넣었어.
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}



        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)
        if self.untie_final_proj:
            proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]
        logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, self.num_classes)] # [[B*T, V]]
        mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices, ~padding_mask).view(-1) # [B*T]
        logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
        target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [target.view(-1)[unmask].long() for target in target_list]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )

        # NLL loss용 noise_logits 계산 (gradient 있음, frame-wise)
        if self._layer1_output is not None and hasattr(self, 'noise_classifier'):
            noise_logits = F.softmax(self.noise_classifier(self._layer1_output), dim=-1)  # [T, B, 4]
        else:
            noise_logits = None

        return x, padding_mask, noise_logits


    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits