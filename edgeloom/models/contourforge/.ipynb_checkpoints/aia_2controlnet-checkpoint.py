# -*- coding: utf-8 -*-
"""
EdgeLoom AIA + Dual ControlNet 实现（不依赖 ddpm_ref_aia / ddim_ref）
- AIA 融合发生在 UNet Block 内部（依赖你导入的 edgeloom.models.modules.diffusionmodules.openaimodel_ssl_aia）
- 两条 ControlNet 分支: ControlNet（含下采样）、ControlNet_latent（stride=1）
- 训练时直接 forward(noisy_latents, timesteps, context, hint, ssl_hint) -> noise_pred
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from einops import rearrange

from edgeloom.models.modules.diffusionmodules.util import (
    conv_nd, linear, zero_module, timestep_embedding,
)
from edgeloom.models.modules.attention import SpatialTransformer
from edgeloom.models.modules.diffusionmodules.openaimodel_ssl_aia import (
    UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
)
from edgeloom.models.util import instantiate_from_config
import torch.nn.functional as F


# ------------------------------
# 1) UNet：AIA在 block 内融合
# ------------------------------
class ControlledUnetModel(UNetModel):
    """
    扩展自 UNetModel。各个 block 内部完成 AIA 融合（接收 control 与 ssl_latent 两路特征）。
    这里负责把 residual list（从 middle 到最浅）逐层喂给 middle/output blocks。
    """
    def __init__(self, *args,
                 freeze_input_blocks: bool = True,
                 freeze_middle_block: bool = True,
                 freeze_output_blocks: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._freeze_in = freeze_input_blocks
        self._freeze_mid = freeze_middle_block
        self._freeze_out = freeze_output_blocks

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor = None,
        context: Optional[torch.Tensor] = None,
        control: Optional[List[torch.Tensor]] = None,
        ssl_latent: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(control, list) and isinstance(ssl_latent, list), \
            "control/ssl_latent 必须是 list，并且顺序与 UNet block 对齐（末尾为 middle）。"
        hs = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # 编码侧（常冻结）
        def encode_path(h):
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            return h

        h = x.type(self.dtype)
        if self._freeze_in:
            with torch.no_grad():
                h = encode_path(h)
        else:
            h = encode_path(h)

        # middle block（AIA 融合）
        if self._freeze_mid:
            with torch.no_grad():
                h = self.middle_block(h, emb, context, control.pop(), ssl_latent.pop())
        else:
            h = self.middle_block(h, emb, context, control.pop(), ssl_latent.pop())

        # 解码侧（逐层 AIA）
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if self._freeze_out:
                with torch.no_grad():
                    h = module(h, emb, context, control.pop(), ssl_latent.pop())
            else:
                h = module(h, emb, context, control.pop(), ssl_latent.pop())

        h = h.type(x.dtype)
        return self.out(h)

    def set_freeze(self, freeze_input: Optional[bool] = None,
                   freeze_middle: Optional[bool] = None,
                   freeze_output: Optional[bool] = None):
        if freeze_input is not None:
            self._freeze_in = freeze_input
        if freeze_middle is not None:
            self._freeze_mid = freeze_middle
        if freeze_output is not None:
            self._freeze_out = freeze_output


# ------------------------------
# 2) 两条 ControlNet 分支
# ------------------------------
class _BaseControl(nn.Module):
    def __init__(
        self,
        image_size, in_channels, model_channels, hint_channels, num_res_blocks,
        attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8),
        conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False,
        num_heads=-1, num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=False, resblock_updown=False,
        use_new_attention_order=False, use_spatial_transformer=False,
        transformer_depth=1, context_dim=None, n_embed=None, legacy=True,
        disable_self_attentions=None, num_attention_blocks=None,
        disable_middle_self_attn=False, use_linear_in_transformer=False,
        stride_pack: tuple = (2, 2, 2),
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "use_spatial_transformer=True 需要提供 context_dim"

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks] if isinstance(num_res_blocks, int) else num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # stem
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # hint/cond 编码塔
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=stride_pack[0]), nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=stride_pack[1]), nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1), nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=stride_pack[2]), nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        # 主干（用于生成多尺度 residual，经 zero_conv 对齐通道）
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels,
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        heads = num_heads
                        dim_head = ch // heads
                    else:
                        heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // heads if use_spatial_transformer else num_head_channels

                    disabled_sa = False if disable_self_attentions is None else disable_self_attentions[level]
                    if (num_attention_blocks is None) or (nr < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch, use_checkpoint=use_checkpoint,
                                num_heads=heads, num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm, down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2

        # middle
        if num_head_channels == -1:
            heads = num_heads
            dim_head = ch // heads
        else:
            heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=heads,
                num_head_channels=dim_head, use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs) -> List[torch.Tensor]:
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs: List[torch.Tensor] = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                # ------------ 关键修改：空间对齐 ------------
                if guided_hint.shape[-2:] != h.shape[-2:]:
                    guided_hint = F.interpolate(
                        guided_hint, size=h.shape[-2:], mode="bilinear", align_corners=False
                    )
                # -----------------------------------------
                h = h + guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)

            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs


class ControlNet(_BaseControl):
    """ 含下采样的 ControlNet（适合边缘/引导） """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, stride_pack=(2, 2, 2), **kwargs)


class ControlNet_latent(_BaseControl):
    """ 不下采样（stride=1）的 ControlNet（适合参考/语义特征） """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, stride_pack=(1, 1, 1), **kwargs)



class ControlAIA_System(nn.Module):
    """
    - AIA UNet（ControlledUnetModel） + 两路 ControlNet
    - 训练接口：forward(noisy_latents, timesteps, context, hint, ssl_hint) -> noise_pred
    - 可选冻结：仅训练两路 ControlNet，或微调 UNet 的 AIA block
    """
    def __init__(
        self,
        unet_config: Dict[str, Any],
        control_stage_config: Dict[str, Any],      # ControlNet（edge/hint）
        ssl_stage_config: Dict[str, Any],          # ControlNet_latent（ref/ssl）
        freeze_unet_in: bool = True,
        freeze_unet_mid: bool = False,
        freeze_unet_out: bool = False,
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        # 1) AIA UNet
        self.unet: ControlledUnetModel = instantiate_from_config(unet_config)
        assert isinstance(self.unet, ControlledUnetModel), "unet_config.target 必须是 ControlledUnetModel"
        self.unet.set_freeze(
            freeze_input=freeze_unet_in,
            freeze_middle=freeze_unet_mid,
            freeze_output=freeze_unet_out,
        )

        # 2) 两个 ControlNet 分支
        self.control_model: _BaseControl = instantiate_from_config(control_stage_config)
        self.ssl_stage_model: _BaseControl = instantiate_from_config(ssl_stage_config)

        # 3) 融合尺度
        self.control_scales = [1.0] * 13
        self.learning_rate = learning_rate

        # 默认：只训练两条 ControlNet
        for p in self.unet.parameters():
            p.requires_grad = not (freeze_unet_in and freeze_unet_mid and freeze_unet_out)
        for p in self.control_model.parameters():
            p.requires_grad = True
        for p in self.ssl_stage_model.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def set_control_scales_uniform(self, s: float):
        self.control_scales = [s for _ in range(len(self.control_scales))]

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        hint: torch.Tensor,
        ssl_hint: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练/推理一步：
        - 两条 ControlNet 生成多尺度 residual list（末尾为 middle）
        - AIA UNet 在各 block 内部融合，输出噪声预测
        """
        # edge/hint 分支
        control = self.control_model(
            x=noisy_latents, hint=hint, timesteps=timesteps, context=encoder_hidden_states
        )
        # 参考/语义 分支
        ssl_res = self.ssl_stage_model(
            x=noisy_latents, hint=ssl_hint, timesteps=timesteps, context=encoder_hidden_states
        )

        # 对齐 scale 长度
        n = len(control)
        scales = self.control_scales if len(self.control_scales) == n else [1.0] * n
        control = [c * s for c, s in zip(control, scales)]
        ssl_res = [c * s for c, s in zip(ssl_res, scales)]

        # AIA 融合在 UNet 内部完成
        noise_pred = self.unet(
            noisy_latents, timesteps, context=encoder_hidden_states,
            control=control, ssl_latent=ssl_res
        )
        return noise_pred

    # 方便你直接拿来用的优化器构造（可选）
    def configure_optimizers(self):
        params = []
        params += [p for p in self.control_model.parameters() if p.requires_grad]
        params += [p for p in self.ssl_stage_model.parameters() if p.requires_grad]
        params += [p for p in self.unet.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.learning_rate)

    # 低显存切换（可选）
    def low_vram_shift(self, is_diffusing: bool):
        if is_diffusing:
            self.unet = self.unet.cuda()
            self.control_model = self.control_model.cpu()
            self.ssl_stage_model = self.ssl_stage_model.cuda()
        else:
            self.unet = self.unet.cpu()
            self.control_model = self.control_model.cuda()
