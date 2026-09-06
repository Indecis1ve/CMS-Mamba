"""Bidirectional multimodal Mamba with paper-aligned MCSSM control."""

import math

import torch
from torch import nn

from models.mamba_nets.selective_scan_interface import selective_scan_fn
from models.missingness import MissingnessConditionedStep


class Mamba(nn.Module):
    """Two-stream bidirectional selective scan used by TC-Mamba.

    Both streams receive the same branch-specific missingness indicator, but
    own independent MCSSM gates and SSM input projections.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=True,
        init_layer_scale=None,
    ):
        super().__init__()
        if bimamba_type != "v2":
            raise ValueError("multimodal MCSSM Mamba requires bimamba_type='v2'")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.use_fast_path = False
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.a_in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.v_in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.a_conv1d = self._depthwise_conv(conv_bias, factory_kwargs)
        self.v_conv1d = self._depthwise_conv(conv_bias, factory_kwargs)
        self.a_conv1d_b = self._depthwise_conv(conv_bias, factory_kwargs)
        self.v_conv1d_b = self._depthwise_conv(conv_bias, factory_kwargs)
        self.act = nn.SiLU()

        projection_dim = self.dt_rank + self.d_state * 2
        self.a_x_proj = nn.Linear(
            self.d_inner, projection_dim, bias=False, **factory_kwargs
        )
        self.v_x_proj = nn.Linear(
            self.d_inner, projection_dim, bias=False, **factory_kwargs
        )
        self.a_x_proj_b = nn.Linear(
            self.d_inner, projection_dim, bias=False, **factory_kwargs
        )
        self.v_x_proj_b = nn.Linear(
            self.d_inner, projection_dim, bias=False, **factory_kwargs
        )
        self.a_dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        self.v_dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        self.a_dt_proj_b = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        self.v_dt_proj_b = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        for projection in (
            self.a_dt_proj,
            self.v_dt_proj,
            self.a_dt_proj_b,
            self.v_dt_proj_b,
        ):
            self._initialize_dt_projection(
                projection,
                dt_min,
                dt_max,
                dt_init,
                dt_scale,
                dt_init_floor,
                factory_kwargs,
            )

        self.a_mcssm = MissingnessConditionedStep(
            self.d_inner,
            mask_dim=2,
            **factory_kwargs,
        )
        self.v_mcssm = MissingnessConditionedStep(
            self.d_inner,
            mask_dim=2,
            **factory_kwargs,
        )

        state_axis = torch.arange(
            1,
            self.d_state + 1,
            dtype=torch.float32,
            device=device,
        )
        state_matrix = state_axis.unsqueeze(0).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(state_matrix))
        self.A_log._no_weight_decay = True
        self.A_b_log = nn.Parameter(torch.log(state_matrix.clone()))
        self.A_b_log._no_weight_decay = True

        self.a_D = self._skip_parameter(device)
        self.v_D = self._skip_parameter(device)
        self.a_D_b = self._skip_parameter(device)
        self.v_D_b = self._skip_parameter(device)

        self.a_out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.v_out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.a_gamma = None
        self.v_gamma = None
        if init_layer_scale is not None:
            self.a_gamma = nn.Parameter(
                float(init_layer_scale) * torch.ones(self.d_model, **factory_kwargs)
            )
            self.v_gamma = nn.Parameter(
                float(init_layer_scale) * torch.ones(self.d_model, **factory_kwargs)
            )
        self.last_mcssm_stats = {}

    def _depthwise_conv(self, conv_bias, factory_kwargs):
        return nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

    def _skip_parameter(self, device):
        parameter = nn.Parameter(torch.ones(self.d_inner, device=device))
        parameter._no_weight_decay = True
        return parameter

    def _initialize_dt_projection(
        self,
        projection,
        dt_min,
        dt_max,
        dt_init,
        dt_scale,
        dt_init_floor,
        factory_kwargs,
    ):
        init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, init_std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -init_std, init_std)
        else:
            raise ValueError(f"unsupported dt_init: {dt_init}")
        initial = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = initial + torch.log(-torch.expm1(-initial))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus)
        projection.bias._no_reinit = True

    def _project_direction(self, x, x_projection, dt_projection, mcssm, indicator):
        # x is [B, D_inner, L]; MCSSM works on [B, L, D_inner].
        features = x.transpose(1, 2).contiguous()
        projected = x_projection(features)
        dt_rank, state_b, state_c = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt_projected = dt_projection(dt_rank)
        mcssm_output = mcssm(features, dt_projected, indicator)
        return (
            mcssm_output.delta.transpose(1, 2).contiguous(),
            state_b.transpose(1, 2).contiguous(),
            state_c.transpose(1, 2).contiguous(),
            mcssm_output,
        )

    @staticmethod
    def _diagnostic(output):
        return {
            "alpha": output.alpha.detach().cpu(),
            "delta": output.delta.detach().cpu(),
            "delta_base": output.delta_base.detach().cpu(),
        }

    def forward(
        self,
        a_hidden_states,
        v_hidden_states,
        missing_indicator,
        a_inference_params=None,
        v_inference_params=None,
    ):
        del a_inference_params, v_inference_params
        if a_hidden_states.shape != v_hidden_states.shape:
            raise ValueError("TC-Mamba stream tensors must have identical shapes")
        if a_hidden_states.ndim != 3:
            raise ValueError("TC-Mamba streams must have shape [B, L, D]")
        batch, sequence_length, _ = a_hidden_states.shape
        expected_mask_shape = (batch, sequence_length, 2)
        if missing_indicator.shape != expected_mask_shape:
            raise ValueError(
                f"missing indicator must have shape {expected_mask_shape}, "
                f"got {missing_indicator.shape}"
            )
        indicator = missing_indicator.to(
            device=a_hidden_states.device,
            dtype=a_hidden_states.dtype,
        )

        a_xz = self.a_in_proj(a_hidden_states).transpose(1, 2).contiguous()
        v_xz = self.v_in_proj(v_hidden_states).transpose(1, 2).contiguous()
        a_x, a_z = a_xz.chunk(2, dim=1)
        v_x, v_z = v_xz.chunk(2, dim=1)
        a_x = self.act(self.a_conv1d(a_x)[..., :sequence_length])
        v_x = self.act(self.v_conv1d(v_x)[..., :sequence_length])

        a_dt, a_b, a_c, a_mcssm_f = self._project_direction(
            a_x, self.a_x_proj, self.a_dt_proj, self.a_mcssm, indicator
        )
        v_dt, v_b, v_c, v_mcssm_f = self._project_direction(
            v_x, self.v_x_proj, self.v_dt_proj, self.v_mcssm, indicator
        )

        a_xz_b = a_xz.flip(-1)
        v_xz_b = v_xz.flip(-1)
        a_x_b, a_z_b = a_xz_b.chunk(2, dim=1)
        v_x_b, v_z_b = v_xz_b.chunk(2, dim=1)
        a_x_b = self.act(self.a_conv1d_b(a_x_b)[..., :sequence_length])
        v_x_b = self.act(self.v_conv1d_b(v_x_b)[..., :sequence_length])
        indicator_b = indicator.flip(1)
        a_dt_b, a_b_b, a_c_b, a_mcssm_b = self._project_direction(
            a_x_b, self.a_x_proj_b, self.a_dt_proj_b, self.a_mcssm, indicator_b
        )
        v_dt_b, v_b_b, v_c_b, v_mcssm_b = self._project_direction(
            v_x_b, self.v_x_proj_b, self.v_dt_proj_b, self.v_mcssm, indicator_b
        )

        state_a = -torch.exp(self.A_log.float())
        state_b = -torch.exp(self.A_b_log.float())
        a_y = selective_scan_fn(
            a_x,
            a_dt,
            state_a,
            a_b,
            a_c,
            self.a_D.float(),
            z=a_z,
            delta_bias=None,
            delta_softplus=False,
        )
        v_y = selective_scan_fn(
            v_x,
            v_dt,
            state_a,
            v_b,
            v_c,
            self.v_D.float(),
            z=v_z,
            delta_bias=None,
            delta_softplus=False,
        )
        a_y_b = selective_scan_fn(
            a_x_b,
            a_dt_b,
            state_b,
            a_b_b,
            a_c_b,
            self.a_D_b.float(),
            z=a_z_b,
            delta_bias=None,
            delta_softplus=False,
        )
        v_y_b = selective_scan_fn(
            v_x_b,
            v_dt_b,
            state_b,
            v_b_b,
            v_c_b,
            self.v_D_b.float(),
            z=v_z_b,
            delta_bias=None,
            delta_softplus=False,
        )

        a_out = self.a_out_proj((a_y + a_y_b.flip(-1)).transpose(1, 2))
        v_out = self.v_out_proj((v_y + v_y_b.flip(-1)).transpose(1, 2))
        if self.a_gamma is not None:
            a_out = a_out * self.a_gamma
            v_out = v_out * self.v_gamma

        self.last_mcssm_stats = {
            "a_forward": self._diagnostic(a_mcssm_f),
            "a_backward": self._diagnostic(a_mcssm_b),
            "v_forward": self._diagnostic(v_mcssm_f),
            "v_backward": self._diagnostic(v_mcssm_b),
        }
        return a_out, v_out
