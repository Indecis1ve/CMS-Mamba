"""Paper-aligned CMS-Mamba model."""

import torch
from torch import nn

from models.bert import BertTextEncoder
from models.mamba import Crossattn, TCMamba, TQMamba
from models.missingness import apply_missing_token
from models.tmm import EnhanceSubNet


class CMSMamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_args = args["model"]
        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers="bert",
            pretrained=model_args["feature_extractor"]["bert_pretrained"],
        )

        vision_dim = int(model_args["tmm"]["input_dim"][1])
        audio_dim = int(model_args["tmm"]["input_dim"][2])
        self.v_mask_token = nn.Parameter(torch.empty(1, 1, vision_dim))
        self.a_mask_token = nn.Parameter(torch.empty(1, 1, audio_dim))
        nn.init.normal_(self.v_mask_token, mean=0.0, std=0.02)
        nn.init.normal_(self.a_mask_token, mean=0.0, std=0.02)

        self.text_modality_mixup = EnhanceSubNet(
            input_length=model_args["tmm"]["input_length"],
            input_dim=model_args["tmm"]["input_dim"],
            hidden_dim=model_args["tmm"]["hidden_dim"],
        )

        tc_args = model_args["tc_mamba"]
        self.text_based_context_mamba = TCMamba(
            num_layers=tc_args["num_layers"],
            d_model=tc_args["d_model"],
            d_ffn=tc_args.get("d_ffn", tc_args["d_model"] * 4),
            activation=tc_args["activation"],
            dropout=tc_args["dropout"],
            causal=tc_args["causal"],
            mamba_config=dict(tc_args["mamba_config"]),
        )

        tq_args = model_args["tq_mamba"]
        self.text_guided_attention = Crossattn(
            num_heads=tq_args["attn_heads"],
            d_model=tq_args["d_model"],
            modal_dim=2 * tq_args["d_model"],
            dropout=tq_args["dropout"],
            rope_base=tq_args.get("rope_base", 10000.0),
        )
        self.text_based_query_mamba = TQMamba(
            num_layers=tq_args["num_layers"],
            d_model=tq_args["d_model"],
            d_ffn=tq_args.get("d_ffn", tq_args["d_model"] * 4),
            activation=tq_args["activation"],
            dropout=tq_args["dropout"],
            causal=tq_args["causal"],
            mamba_config=dict(tq_args["mamba_config"]),
        )

        regression_dim = int(model_args["regression"]["input_dim"])
        reconstruction_args = model_args["reconstruction"]
        reconstruction_dropout = float(reconstruction_args.get("dropout", 0.0))
        self.text_reconstructor = nn.Sequential(
            nn.Linear(regression_dim, regression_dim),
            nn.ReLU(),
            nn.Dropout(reconstruction_dropout),
            nn.Linear(
                regression_dim,
                int(reconstruction_args.get("output_dim", 768)),
            ),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.norm_lock = nn.LayerNorm(regression_dim)
        self.output = nn.Linear(
            regression_dim,
            int(model_args["regression"]["out_dim"]),
        )

    @staticmethod
    def _validate_missing_mask(name, missing_mask, valid_mask):
        if missing_mask.shape != valid_mask.shape:
            raise ValueError(
                f"{name} missing and validity masks must have identical shapes"
            )
        missing = missing_mask.to(dtype=torch.bool)
        valid = valid_mask.to(dtype=torch.bool)
        if torch.any(missing & ~valid):
            raise ValueError(f"{name} missing mask contains padding positions")

    def forward(
        self,
        incomplete_input,
        automatic_missing_masks,
        valid_masks,
        conditioning_masks=None,
        complete_text=None,
    ):
        """Run CMS-Mamba with input-derived substitution and selected MCSSM masks.

        ``automatic_missing_masks`` always controls LMMT substitution.  The
        optional ``conditioning_masks`` selects the MCSSM signal, allowing a
        reference-conditioned mechanism experiment without leaking reference
        masks into the deployable input-stabilization path.
        """
        vision, audio, language = incomplete_input
        text_auto_missing, audio_auto_missing, vision_auto_missing = (
            automatic_missing_masks
        )
        text_valid, audio_valid, vision_valid = valid_masks
        if conditioning_masks is None:
            conditioning_masks = automatic_missing_masks
        text_missing, audio_missing, vision_missing = conditioning_masks
        self._validate_missing_mask("automatic text", text_auto_missing, text_valid)
        self._validate_missing_mask("automatic audio", audio_auto_missing, audio_valid)
        self._validate_missing_mask(
            "automatic vision", vision_auto_missing, vision_valid
        )
        self._validate_missing_mask("conditioning text", text_missing, text_valid)
        self._validate_missing_mask("conditioning audio", audio_missing, audio_valid)
        self._validate_missing_mask(
            "conditioning vision", vision_missing, vision_valid
        )

        vision_stable = apply_missing_token(
            vision,
            vision_auto_missing,
            vision_valid,
            self.v_mask_token,
        )
        audio_stable = apply_missing_token(
            audio,
            audio_auto_missing,
            audio_valid,
            self.a_mask_token,
        )
        text_encoded = self.bertmodel(language)

        aligned = self.text_modality_mixup(
            text_encoded,
            vision_stable,
            audio_stable,
            text_missing,
            vision_missing,
            audio_missing,
            vision_valid,
            audio_valid,
        )
        audio_out, vision_out, text_at, text_vt = self.text_based_context_mamba(
            aligned.audio,
            aligned.vision,
            aligned.text,
            aligned.text_missing,
            aligned.audio_missing,
            aligned.vision_missing,
        )
        text_query = (text_at + text_vt) / 2
        modal_key_value = torch.cat((vision_out, audio_out), dim=-1)
        attended = self.text_guided_attention(text_query, modal_key_value)
        fused = self.text_based_query_mamba(attended)
        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)
        prediction = self.output(self.norm_lock(pooled))
        output = {"sentiment_preds": prediction}
        if complete_text is not None:
            if complete_text.ndim != 3:
                raise ValueError("complete text input must have shape [B, 3, L]")
            output["reconstructed_text"] = self.text_reconstructor(aligned.text)
            with torch.no_grad():
                output["complete_text_features"] = self.bertmodel(complete_text)
        return output


def build_model(args):
    return CMSMamba(args)
