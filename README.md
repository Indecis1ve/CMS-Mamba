# CMS-Mamba: Missing-Aware State-Space Modeling for Robust Multimodal Sentiment Analysis under Incomplete Observations

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Mamba](https://img.shields.io/badge/Backbone-Mamba-6f42c1)
![License](https://img.shields.io/badge/License-MIT-green)

**Official PyTorch implementation of CMS-Mamba** — a **missing-aware state-space stabilization framework** for robust multimodal sentiment analysis under incomplete observations.

CMS-Mamba is specifically designed for real-world human-centered engineering AI systems where textual, acoustic, and visual modalities can be severely degraded due to sensor failure, camera occlusion, packet loss, privacy masking, feature extraction errors, or unstable edge deployment.

Instead of attempting to reconstruct missing semantics, CMS-Mamba focuses on **survivability and controlled degradation**: it stabilizes the input representation, recurrent state-space dynamics, and final prediction scale so that the model maintains stable fallback behavior even under extreme missingness (η = 1.0).

The framework introduces three synergistic defense mechanisms:
- **LMMT** (Learnable Missing Modality Tokens) — spatial defense against zero-value bias
- **DTF** (Dynamic Time-Freezing) — temporal/state-level missing-aware ODE step-size control
- **RNL** (Representation Normalization Lock) — numerical/prediction-level feature-scale stabilization

---

## 🔥 Key Features & Engineering Contributions

- **Missing-Aware State-Space Stabilization**: Directly addresses zero-value bias, state drift, and feature-magnitude instability in Mamba-based multimodal models under severe degradation.
- **Extreme Robustness**: Supports evaluation from η = 0.0 (complete data) to η = 1.0 (catastrophic simultaneous text-audio-vision missingness) as a stress-test upper bound.
- **Realistic Missingness Protocols**: Evaluated under block missingness, text-heavy corruption, audio/vision-heavy corruption, mixed burst corruption, etc.
- **Long-Sequence Support**: Efficiently handles unaligned sequences up to 500+ frames (e.g., CMU-MOSEI).
- **Edge-Deployment Friendly**: Validated on NVIDIA Jetson AGX Orin with significantly reduced peak VRAM and better scalability.
- **Cross-Lingual Generalization**: Strong performance on both English (CMU-MOSI/MOSEI) and Chinese (CH-SIMS) datasets.
- **Controlled Fallback Behavior**: Does **not** hallucinate missing affective semantics; instead maintains stable uncertainty and numerical reliability.

---

## 🧠 Method Overview (Aligned with Paper)

CMS-Mamba formulates robust multimodal sentiment analysis as a **missing-aware state-space stabilization** problem. The overall architecture implements a three-layer hierarchical defense:

1. **Input-Level Spatial Defense (LMMT)**: Replaces missing acoustic/visual frames with learnable non-zero modality anchors to prevent zero-value bias and manifold collapse in SSM dynamics.
2. **Text-Aware Modality Mixing + State-Level Temporal Defense (DTF in TC-Mamba)**: Uses CTC-inspired alignment and Dynamic Time-Freezing to regulate the effective discretization step size Δ_t of the Mamba ODE according to missingness indicators and feature reliability.
3. **Deep Query Fusion + Prediction-Level Numerical Defense (RNL)**: RoPE-enhanced cross-attention + TQ-Mamba for sequence-level reasoning, followed by Representation Normalization Lock before the regression head.

For the full technical details, please refer to the paper: "Missing-Aware State-Space Modeling for Robust Multimodal Sentiment Analysis under Incomplete Observations".

**Core Principle**: Spatial → Temporal → Numerical stabilization ensures smooth degradation and numerical reliability under incomplete observations.

---

## ✨ Detailed Key Components

### 1. LMMT: Learnable Missing Modality Tokens (Input-Level Stabilization)
- Conventional zero-padding causes zero-value bias and hidden-state attenuation in Mamba's ODE.
- LMMT provides trainable modality-specific anchors for missing audio and visual frames.
- Textual missing tokens use standard [UNK] (while preserving [CLS] and [SEP]).
- This breaks high-dimensional symmetry and supplies stable non-zero energy to downstream SSM layers.

### 2. DTF: Dynamic Time-Freezing (State-Level Stabilization)
- Embedded in TC-Mamba (Text-Context Mamba).
- Adaptively controls the effective ODE discretization step Δ_t = α_t · Δ_base,t.
- Reliability gate α_t + feature-dependent base step Δ_base,t together suppress harmful updates for uninformative inputs while allowing controlled integration when LMMT anchors are present.
- Prevents uncontrolled state drift during long missing segments.

### 3. RNL: Representation Normalization Lock (Prediction-Level Stabilization)
- Applied before the final regression head.
- Uses LayerNorm-style scaling with learnable affine parameters to constrain feature magnitude drift caused by long sequences of low-variance LMMT signals.

### Additional Modules
- **Text-Aware Modality Mixing (TMM)**: CTC-inspired temporal compression/alignment of long audio-visual sequences to text length.
- **RoPE-Enhanced Deep Query Fusion**: Rotary Position Embedding + TQ-Mamba for position-aware multimodal reasoning even under homogeneous fallback representations.
- **Missingness Indicator Propagation**: Aligned missing masks are fed into DTF for precise control.

---

## 📁 Project Structure

```
CMS-Mamba/
├── ckpt/                     # Trained checkpoints
├── configs/                  # YAML configuration files
│   ├── train_mosi.yaml
│   ├── train_mosei.yaml
│   ├── train_sims.yaml
│   ├── eval_mosi.yaml
│   ├── eval_mosei.yaml
│   └── eval_sims.yaml
├── core/                     # Core utilities
│   ├── dataset.py
│   ├── losses.py
│   ├── metric.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── utils.py
├── data/                     # Place processed .pkl files here
├── models/
│   ├── mamba_nets/           # Mamba backbone components
│   ├── basic_layers.py
│   ├── bert.py
│   ├── mamba.py
│   ├── TFMamba.py
│   └── tmm.py
├── environment.yml
├── robust_evaluation.py
├── train.py
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Indecis1ve/CMS-Mamba.git
cd CMS-Mamba
```

### 2. Create Conda environment
```bash
conda env create -f environment.yml
conda activate CMSmamba
```

### 3. Install Mamba-SSM dependencies
```bash
pip install causal-conv1d
pip install mamba-ssm
```

**Note for NVIDIA Jetson / ARM CUDA devices**: Compile `causal-conv1d` and `mamba-ssm` from source for best compatibility.

---

## 📦 Data Preparation

Download and preprocess the following datasets:

- **CMU-MOSI**
- **CMU-MOSEI**
- **CH-SIMS** (Chinese)

Place the processed feature files (`unaligned_50.pkl` recommended) in:
```
data/
├── CMU_MOSI/
│   └── unaligned_50.pkl
├── CMU_MOSEI/
│   └── unaligned_50.pkl
└── CH_SIMS/
    └── unaligned_50.pkl
```

**BERT weights** (offline support):
- English: `./bert-base-uncased/`
- Chinese: `./bert-base-chinese/`

---

## 🚀 Quick Start

### Training
```bash
# CMU-MOSI
python train.py --config_file configs/train_mosi.yaml

# CMU-MOSEI
python train.py --config_file configs/train_mosei.yaml

# CH-SIMS
python train.py --config_file configs/train_sims.yaml
```

Checkpoints are saved to `./ckpt/`.

### Robustness Evaluation
```bash
# CMU-MOSI
python robust_evaluation.py --config_file configs/eval_mosi.yaml

# CMU-MOSEI
python robust_evaluation.py --config_file configs/eval_mosei.yaml

# CH-SIMS
python robust_evaluation.py --config_file configs/eval_sims.yaml
```

Results are saved under `log/results/`.

---

## 📋 Missing-Modality Protocol

Missing rate η controls the proportion of corrupted elements:

| η Value      | Description                                      |
|--------------|--------------------------------------------------|
| η = 0.0      | Complete multimodal input                        |
| 0 < η < 1.0  | Partial/random + structured missingness          |
| η = 1.0      | **Stress-test upper bound** — full simultaneous text/audio/vision missingness |

At η = 1.0:
- Text: non-special tokens → [UNK], [CLS]/[SEP] preserved
- Audio & Vision: zero-padded → replaced by LMMT anchors

**Important**: η=1.0 is a **stress test for system stability**, not for semantic reconstruction. CMS-Mamba maintains controlled fallback priors.

---

## 📊 Experimental Results (Directly from Paper)

### Complete Data Performance (η = 0.0)

**CMU-MOSI**:
- MAE: 0.7496 | Corr: 0.7796 | Acc-2: 83.23% | F1: 82.81%

**CMU-MOSEI**:
- MAE: 0.5536 | Corr: 0.7598 | Acc-2: 85.61% | F1: 85.56%

### Stress-Test Upper Bound (η = 1.0)

**CMU-MOSEI**:
- MAE reduced from **0.9485** (TF-Mamba) to **0.8389** (CMS-Mamba)

### Average Robustness (η ∈ [0.0, 0.9])

CMS-Mamba outperforms TF-Mamba on most averaged metrics across missing rates.

### Realistic Missingness Patterns

Detailed results available in the paper (Table for block missingness, text missing, A+V missing, text-heavy, A/V-heavy, mixed burst, etc.). CMS-Mamba shows particularly strong gains under text-corrupted and block-corrupted scenarios.

### Edge Deployment (NVIDIA Jetson AGX Orin)

- **66.75%** reduction in peak VRAM at batch size 16
- Remains executable at batch size 32 (baseline OOM)
- Superior throughput and stable performance under missingness

Full results, ablation studies, dynamics visualization, and cross-lingual results (CH-SIMS) are provided in the paper.

---

## 🧪 Ablation Study Highlights (CMU-MOSEI)

- Removing LMMT severely hurts extreme missingness performance
- Removing DTF increases regression error under η=1.0
- Removing RNL causes feature-scale drift
- Full CMS-Mamba achieves best trade-off

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{hu_cmsmamba,
  title={Missing-Aware State-Space Modeling for Robust Multimodal Sentiment Analysis under Incomplete Observations},
  author={Hu, Jie and Li, Ming},
  note={Under review},
  year={2026}
}
```

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

This work builds upon Mamba, TF-Mamba, and the multimodal affective computing community. Supported by relevant Hubei Province programs.

---

**For detailed methodology, proofs, visualizations, and extensive experimental analysis, please read the full paper.**

---

**Ready for download and use!** This README has been comprehensively updated to be fully consistent with the paper content while remaining practical and user-friendly.
