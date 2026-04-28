import torch
from torch import nn
from models.bert import BertTextEncoder
from einops import rearrange, repeat
from models.tmm import EnhanceSubNet
from models.mamba import TCMamba, TQMamba, Crossattn

class TFMamba(nn.Module):
    def __init__(self, args):
        super(TFMamba, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        # ==========================================================
        # Learnable Missing Modality Tokens (LMMT)
        vision_dim = args['model']['tmm']['input_dim'][1] # 默认 20
        audio_dim = args['model']['tmm']['input_dim'][2]  # 默认 5
        
        self.v_mask_token = nn.Parameter(torch.randn(1, 1, vision_dim) * 0.02)
        self.a_mask_token = nn.Parameter(torch.randn(1, 1, audio_dim) * 0.02)
        # ==========================================================

        # TME
        self.text_modality_mixup = EnhanceSubNet(
            input_length=args['model']['tmm']['input_length'],
            input_dim=args['model']['tmm']['input_dim'],
            hidden_dim=args['model']['tmm']['hidden_dim']
        )
        
        # feature reconstruction
        self.recon_text_low = nn.Sequential(
            nn.Linear(args['model']['tmr']['input_dim_high'], args['model']['tmr']['input_dim_high']),
            nn.ReLU(),
            nn.Dropout(args['model']['tmr']['dropout']),
            nn.Linear(args['model']['tmr']['input_dim_high'], args['model']['tmr']['input_dim_low'])
        )
        
        # TC-Mamba
        self.text_based_context_mamba = TCMamba(
            num_layers=args['model']['tc_mamba']['num_layers'],
            d_model=args['model']['tc_mamba']['d_model'],
            d_ffn=args['model']['tc_mamba']['d_model'] * 4,
            activation=args['model']['tc_mamba']['activation'],
            dropout=args['model']['tc_mamba']['dropout'],
            causal=args['model']['tc_mamba']['causal'],
            mamba_config=args['model']['tc_mamba']['mamba_config']
        )
        
        # TQ-Mamba
        self.text_guided_attention = Crossattn(
            num_heads=args['model']['tq_mamba']['attn_heads'],
            d_model=args['model']['tq_mamba']['d_model'],
        )
        self.text_based_query_mamba = TQMamba(
            num_layers=args['model']['tq_mamba']['num_layers'],
            d_model=args['model']['tq_mamba']['d_model'],
            d_ffn=args['model']['tq_mamba']['d_model'] * 4,
            activation=args['model']['tq_mamba']['activation'],
            dropout=args['model']['tq_mamba']['dropout'],
            causal=args['model']['tq_mamba']['causal'],
            mamba_config=args['model']['tq_mamba']['mamba_config']
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        # 特征归一化锁 (Representation Normalization Lock)
        # 防止极端缺失率下 Mamba 连续积分导致的数值膨胀，稳定回归头 MAE
        self.norm_lock = nn.LayerNorm(args['model']['regression']['input_dim'])
        self.output = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['out_dim'])

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        # ==========================================================
        # [动态拦截拦截纯 0 噪声] 
        # 检测哪些帧是完全缺失的 (全为 0)
        v_is_missing = (vision_m == 0).all(dim=-1, keepdim=True) # [B, L, 1]
        a_is_missing = (audio_m == 0).all(dim=-1, keepdim=True)  # [B, L, 1]

        # 用可学习的 Token 替换掉纯 0 的帧
        h_0_v = torch.where(v_is_missing, self.v_mask_token, vision_m)
        h_0_a = torch.where(a_is_missing, self.a_mask_token, audio_m)
        # ==========================================================

        h_0_t = self.bertmodel(language_m)
        
        h_tmm_t, h_tmm_v, h_tmm_a = self.text_modality_mixup(h_0_t, h_0_v, h_0_a)
        
        # 状态冻结 Mamba
        h_tc_mamba_a, h_tc_mamba_v, h_tc_mamba_t = self.text_based_context_mamba(h_tmm_a, h_tmm_v, h_tmm_t)

        h_tm_attn = self.text_guided_attention(h_tc_mamba_t, torch.cat([h_tc_mamba_a, h_tc_mamba_v], dim=1))
        h_tm_mamba = self.text_based_query_mamba(h_tm_attn)

        h_m_pool = self.pool(h_tm_mamba.permute(0, 2, 1)).squeeze(-1)
        # 过一次 LayerNorm，把膨胀的特征强行拉回标准分布
        h_m_pool_locked = self.norm_lock(h_m_pool) 
        output = self.output(h_m_pool_locked)

        rec_text_feats, com_text_feats = None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            h_t_o = self.bertmodel(language)
            text_recon_low = self.recon_text_low(h_tmm_t)
            rec_text_feats = [text_recon_low]
            com_text_feats = [h_t_o]

        return {'sentiment_preds': output,
                'rec_text': rec_text_feats,
                'complete_text': com_text_feats}

def build_model(args):
    return TFMamba(args)