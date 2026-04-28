'''
Copied and modified from XX
'''

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from models.mamba_nets.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, \
    mamba_inner_fn_no_out_proj

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True, layer_idx=None, device=None, dtype=None, bimamba_type="none", if_devide_out=True, init_layer_scale=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = False
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        assert bimamba_type == 'v2'

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.a_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
            self.v_gammagamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.a_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.v_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.a_conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.v_conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.activation = "silu"
        self.act = nn.SiLU()

        self.a_x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.a_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.v_x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.v_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.a_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.v_dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.a_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.v_dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        a_dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        a_inv_dt = a_dt + torch.log(-torch.expm1(-a_dt))
        with torch.no_grad():
            self.a_dt_proj.bias.copy_(a_inv_dt)
        self.a_dt_proj.bias._no_reinit = True

        v_dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        v_inv_dt = v_dt + torch.log(-torch.expm1(-v_dt))
        with torch.no_grad():
            self.v_dt_proj.bias.copy_(v_inv_dt)
        self.v_dt_proj.bias._no_reinit = True

        # ==========================================================
        # 用单维度标量来评估自己当前帧是否是噪声/缺失
        self.a_time_gate = nn.Linear(self.d_inner, 1, bias=True, **factory_kwargs)
        self.v_time_gate = nn.Linear(self.d_inner, 1, bias=True, **factory_kwargs)
        
        with torch.no_grad():
            self.a_time_gate.bias.fill_(2.0)
            self.v_time_gate.bias.fill_(2.0)
        # ==========================================================

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.a_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.a_D._no_weight_decay = True
        self.v_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.v_D._no_weight_decay = True

        if bimamba_type == "v2":
            A_b = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.A_b_log._no_weight_decay = True

            self.a_conv1d_b = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
            self.v_conv1d_b = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)

            self.a_x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.a_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.v_x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.v_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.a_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.a_D_b._no_weight_decay = True
            self.v_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.v_D_b._no_weight_decay = True

        self.a_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.v_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, a_hidden_states, v_hidden_states, a_inference_params=None, v_inference_params=None):
        batch, seqlen, dim = a_hidden_states.shape
        
        a_xz = rearrange(self.a_in_proj.weight @ rearrange(a_hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.a_in_proj.bias is not None:
            a_xz = a_xz + rearrange(self.a_in_proj.bias.to(dtype=a_xz.dtype), "d -> d 1")
            
        v_xz = rearrange(self.v_in_proj.weight @ rearrange(v_hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.v_in_proj.bias is not None:
            v_xz = v_xz + rearrange(self.v_in_proj.bias.to(dtype=v_xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        a_x, a_z = a_xz.chunk(2, dim=1)
        v_x, v_z = v_xz.chunk(2, dim=1)

        a_x = self.act(self.a_conv1d(a_x)[..., :seqlen])
        v_x = self.act(self.v_conv1d(v_x)[..., :seqlen])

        # ==========================================================
        # Dynamic Time-Freezing (DTF) Mechanism
        a_x_flat = rearrange(a_x, "b d l -> (b l) d")
        v_x_flat = rearrange(v_x, "b d l -> (b l) d")

        a_x_dbl = self.a_x_proj(a_x_flat)
        a_dt, a_B, a_C = torch.split(a_x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        v_x_dbl = self.v_x_proj(v_x_flat)
        v_dt, v_B, v_C = torch.split(v_x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 1. 评估自身健康状态
        a_health_gate = torch.sigmoid(self.a_time_gate(a_x_flat))
        v_health_gate = torch.sigmoid(self.v_time_gate(v_x_flat))

        # 2. 投影出原始步长 dt_proj (包含 bias)
        a_dt_proj = (self.a_dt_proj.weight @ a_dt.t()) + rearrange(self.a_dt_proj.bias.to(dtype=a_dt.dtype), "d -> d 1")
        v_dt_proj = (self.v_dt_proj.weight @ v_dt.t()) + rearrange(self.v_dt_proj.bias.to(dtype=v_dt.dtype), "d -> d 1")

        # 3. 计算 Base Step (通过 Softplus，严格对应公式: \Delta_base = ln(1 + exp(\Delta_proj)))
        a_dt_base = F.softplus(a_dt_proj)
        v_dt_base = F.softplus(v_dt_proj)

        # 4. Multiplicative Gating (乘法门控，严格对应公式: \Delta_t = \alpha_t * \Delta_base)
        a_dt_frozen = a_dt_base * a_health_gate.t()
        v_dt_frozen = v_dt_base * v_health_gate.t()

        self.vis_alpha_a = a_health_gate.detach().cpu()
        self.vis_alpha_v = v_health_gate.detach().cpu()
        self.vis_a_dt = a_dt_frozen.detach().cpu() # 直接抓取最终 frozen 后的实际步长
        self.vis_v_dt = v_dt_frozen.detach().cpu()


        a_dt = rearrange(a_dt_frozen, "d (b l) -> b d l", l=seqlen)
        a_B = rearrange(a_B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        a_C = rearrange(a_C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        v_dt = rearrange(v_dt_frozen, "d (b l) -> b d l", l=seqlen)
        v_B = rearrange(v_B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        v_C = rearrange(v_C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # ==========================================================


        a_y = selective_scan_fn(a_x, a_dt, A, a_B, a_C, self.a_D.float(), z=a_z, delta_bias=None, delta_softplus=False)
        v_y = selective_scan_fn(v_x, v_dt, A, v_B, v_C, self.v_D.float(), z=v_z, delta_bias=None, delta_softplus=False)

        a_y = rearrange(a_y, "b d l -> b l d")
        a_out = self.a_out_proj(a_y)

        v_y = rearrange(v_y, "b d l -> b l d")
        v_out = self.v_out_proj(v_y)

        return a_out, v_out
