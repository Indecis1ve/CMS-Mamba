"""TC-Mamba, TQ-Mamba, and paper-aligned cross-modal fusion wrappers."""

import torch
from torch import nn

from mamba_ssm import Mamba as UniMamba

from models.mamba_nets.attention import RotaryKeyCrossAttention
from models.mamba_nets.bimamba import Mamba as BiMamba
from models.mamba_nets.mm_bimamba import Mamba as MMBiMamba


class MMMambaEncoderLayer(nn.Module):
    """A two-stream TC layer controlled by one branch missingness indicator."""

    def __init__(
        self,
        d_model,
        d_ffn,
        activation="Swish",
        dropout=0.1,
        causal=False,
        mamba_config=None,
    ):
        super().__init__()
        del d_ffn, activation, dropout
        if mamba_config is None:
            raise ValueError("TC-Mamba requires a mamba_config")
        config = dict(mamba_config)
        bidirectional = bool(config.pop("bidirectional", True))
        if causal or not bidirectional:
            raise ValueError(
                "paper-aligned TC-Mamba requires bidirectional selective scans"
            )
        self.mamba = MMBiMamba(
            d_model=d_model,
            bimamba_type="v2",
            **config,
        )
        self.left_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.right_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        left,
        right,
        missing_indicator,
        left_inference_params=None,
        right_inference_params=None,
    ):
        left_delta, right_delta = self.mamba(
            left,
            right,
            missing_indicator,
            left_inference_params,
            right_inference_params,
        )
        return (
            left + self.left_norm(left_delta),
            right + self.right_norm(right_delta),
        )


class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation="Swish",
        dropout=0.1,
        causal=False,
        mamba_config=None,
    ):
        super().__init__()
        del d_ffn, activation, dropout
        if mamba_config is None:
            raise ValueError("TQ-Mamba requires a mamba_config")
        config = dict(mamba_config)
        bidirectional = bool(config.pop("bidirectional", True))
        if causal or not bidirectional:
            self.mamba = UniMamba(d_model=d_model, **config)
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type="v2",
                **config,
            )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, inference_params=None):
        return x + self.norm(self.mamba(x, inference_params))


class TCMamba(nn.Module):
    """Parallel AT and VT branches that retain separate text contexts."""

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn=1024,
        activation="Swish",
        dropout=0.1,
        causal=False,
        mamba_config=None,
    ):
        super().__init__()
        layer_arguments = {
            "d_model": d_model,
            "d_ffn": d_ffn,
            "dropout": dropout,
            "activation": activation,
            "causal": causal,
            "mamba_config": mamba_config,
        }
        self.at_mamba_layers = nn.ModuleList(
            MMMambaEncoderLayer(**layer_arguments) for _ in range(num_layers)
        )
        self.vt_mamba_layers = nn.ModuleList(
            MMMambaEncoderLayer(**layer_arguments) for _ in range(num_layers)
        )

    def forward(
        self,
        audio,
        vision,
        text,
        text_missing,
        audio_missing,
        vision_missing,
        audio_inference_params=None,
        vision_inference_params=None,
        text_inference_params=None,
    ):
        if not (
            text_missing.shape
            == audio_missing.shape
            == vision_missing.shape
            == text.shape[:2]
        ):
            raise ValueError("TC-Mamba aligned missing masks must match text time")
        audio_out = audio
        vision_out = vision
        text_at = text
        text_vt = text
        at_indicator = torch.stack((text_missing, audio_missing), dim=-1)
        vt_indicator = torch.stack((text_missing, vision_missing), dim=-1)

        for at_layer, vt_layer in zip(
            self.at_mamba_layers,
            self.vt_mamba_layers,
        ):
            audio_out, text_at = at_layer(
                audio_out,
                text_at,
                at_indicator,
                audio_inference_params,
                text_inference_params,
            )
            vision_out, text_vt = vt_layer(
                vision_out,
                text_vt,
                vt_indicator,
                vision_inference_params,
                text_inference_params,
            )
        return audio_out, vision_out, text_at, text_vt


class TQMamba(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn=1024,
        activation="Swish",
        dropout=0.1,
        causal=False,
        mamba_config=None,
    ):
        super().__init__()
        self.mamba_layers = nn.ModuleList(
            MambaEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            )
            for _ in range(num_layers)
        )

    def forward(self, x, inference_params=None):
        output = x
        for layer in self.mamba_layers:
            output = layer(output, inference_params=inference_params)
        return output


class Crossattn(nn.Module):
    def __init__(
        self,
        num_heads,
        d_model,
        modal_dim=None,
        dropout=0.0,
        rope_base=10000.0,
    ):
        super().__init__()
        self.cross_attention = RotaryKeyCrossAttention(
            query_dim=d_model,
            modal_dim=2 * d_model if modal_dim is None else modal_dim,
            heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )

    def forward(self, text_query, modal_key_value):
        return self.cross_attention(text_query, modal_key_value)
