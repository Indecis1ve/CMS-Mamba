# CMS-Mamba: A Context-Maintained Surviving Framework for Robust Multimodal Sentiment Analysis under Extreme Missingness

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Mamba](https://img.shields.io/badge/Backbone-Mamba-6f42c1)
![License](https://img.shields.io/badge/License-MIT-green)

Official PyTorch implementation of **CMS-Mamba**, a **Context-Maintained Surviving Mamba** framework for robust multimodal sentiment analysis under extreme modality missingness.

CMS-Mamba targets real-world multimodal affective computing scenarios where text, audio, and visual streams may be partially or completely corrupted due to sensor failure, camera occlusion, packet loss, privacy masking, or unstable edge deployment.

Instead of relying on reconstruction or simple zero-padding, CMS-Mamba introduces a hierarchical defense system to stabilize Mamba-based multimodal fusion under severe data degradation.

---

## 🔥 Highlights

- **Extreme Missing-Modality Robustness**  
  Supports robustness evaluation from missing rate `η = 0.0` to `η = 1.0`, including catastrophic simultaneous text-audio-vision corruption.

- **Spatiotemporal Orthogonal Defense**  
  A hierarchical defense framework consisting of:
  - **LMMT**: Learnable Missing Modality Tokens
  - **DTF**: Dynamic Time-Freezing
  - **RNL**: Representation Normalization Lock

- **State-Space Stability under Missingness**  
  Addresses zero-value bias, state drift, and feature-magnitude instability in Mamba-based multimodal sentiment analysis.

- **Text-Aware Long-Sequence Modeling**  
  Uses CTC-inspired text-aware modality mixing to align long acoustic and visual streams with the text sequence, especially on CMU-MOSEI where sequences can reach 500 frames.

- **Edge Deployment Friendly**  
  Verified on NVIDIA Jetson AGX Orin. CMS-Mamba reduces measured peak CUDA memory footprint and avoids the OOM failure encountered by the reproduced TF-Mamba baseline.

- **Cross-Lingual Dataset Support**  
  Supports English datasets **CMU-MOSI**, **CMU-MOSEI**, and the Chinese multimodal sentiment dataset **CH-SIMS**.

---

## 🧠 Method Overview

```text
Input Modalities
 ├── Text
 ├── Audio
 └── Vision
        │
        ▼
Spatial Defense
 └── LMMT: Learnable Missing Modality Tokens
        │
        ▼
Text-Aware Modality Mixing
 └── CTC-inspired temporal pseudo-alignment
        │
        ▼
Temporal Defense
 └── TC-Mamba with Dynamic Time-Freezing
        │
        ▼
Deep Query Fusion
 └── RoPE-enhanced cross-attention + TQ-Mamba
        │
        ▼
Numerical Defense
 └── RNL: Representation Normalization Lock
        │
        ▼
Sentiment Prediction
```

---

## ✨ Key Components

### LMMT: Learnable Missing Modality Tokens

LMMT replaces zero-padded missing audio and visual frames with trainable modality-specific tokens.

This prevents continuous acoustic and visual streams from collapsing into high-dimensional zero vectors. For text, missing non-special tokens are replaced by `[UNK]`, while `[CLS]` and `[SEP]` are preserved to maintain valid BERT sentence boundaries.

### DTF: Dynamic Time-Freezing

DTF is embedded into the Mamba state-space engine as a missing-aware step-size controller.

It regulates the effective discretization step size of the Mamba ODE system. Severely uninformative inputs can drive the state update toward a near-frozen regime, while LMMT-stabilized missing frames can still be integrated through a small and stable positive step.

### RNL: Representation Normalization Lock

RNL is applied before the final regression head.

It suppresses feature-scale drift and numerical divergence caused by long-term processing of low-variance or missing-modality representations.

---

## 📁 Directory Structure

```text
CMS-Mamba/
├── ckpt/
├── configs/
│   ├── eval_mosei.yaml
│   ├── eval_mosi.yaml
│   ├── eval_sims.yaml
│   ├── train_mosei.yaml
│   ├── train_mosi.yaml
│   └── train_sims.yaml
├── core/
│   ├── dataset.py
│   ├── losses.py
│   ├── metric.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── utils.py
├── data/
├── models/
│   ├── mamba_nets/
│   │   ├── attention.py
│   │   ├── bimamba.py
│   │   ├── mamba_blocks.py
│   │   ├── mm_bimamba.py
│   │   └── selective_scan_interface.py
│   ├── basic_layers.py
│   ├── bert.py
│   ├── mamba.py
│   ├── TFMamba.py
│   └── tmm.py
├── environment.yml
├── robust_evaluation.py
└── train.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Indecis1ve/CMS-Mamba.git
cd CMS-Mamba
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate CMSmamba
```

### 3. Install Mamba dependencies

```bash
pip install causal-conv1d
pip install mamba-ssm
```

> For NVIDIA Jetson AGX Orin, Jetson Orin NX, or other ARM-based CUDA devices, compiling `causal-conv1d` and `mamba-ssm` from source is recommended to avoid binary compatibility issues.

---

## 📦 Data Preparation

Please manually download and preprocess the multimodal feature files for each dataset.

Place the processed `.pkl` files according to the paths configured in `configs/*.yaml`.

Example:

```text
data/
├── CMU_MOSI/
│   └── unaligned_50.pkl
├── CMU_MOSEI/
│   └── unaligned_50.pkl
└── CH_SIMS/
    └── unaligned_50.pkl
```

The datasets used in this project are:

- **CMU-MOSI**
- **CMU-MOSEI**
- **CH-SIMS**

CMU-MOSI and CMU-MOSEI can be obtained through the CMU Multimodal SDK. CH-SIMS is available from its official repository.

---

## 🤗 Offline BERT Weights

CMS-Mamba supports local BERT loading for offline HPC clusters and edge devices.

For English datasets:

```text
./bert-base-uncased
```

For Chinese datasets:

```text
./bert-base-chinese
```

Expected structure:

```text
CMS-Mamba/
├── bert-base-uncased/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
└── bert-base-chinese/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── vocab.txt
```

---

## 🚀 Quick Start

### Train on CMU-MOSI

```bash
python train.py --config_file configs/train_mosi.yaml
```

### Train on CMU-MOSEI

```bash
python train.py --config_file configs/train_mosei.yaml
```

### Train on CH-SIMS

```bash
python train.py --config_file configs/train_sims.yaml
```

Checkpoints are saved under:

```text
ckpt/
```

---

## 🧪 Robustness Evaluation

CMS-Mamba supports evaluation across missing rates from `η = 0.0` to `η = 1.0`.

### Evaluate on CMU-MOSI

```bash
python robust_evaluation.py --config_file configs/eval_mosi.yaml
```

### Evaluate on CMU-MOSEI

```bash
python robust_evaluation.py --config_file configs/eval_mosei.yaml
```

### Evaluate on CH-SIMS

```bash
python robust_evaluation.py --config_file configs/eval_sims.yaml
```

Results are saved in:

```text
log/results/
```

---

## 🔬 Missing-Modality Protocol

The missing rate is denoted as `η`.

| Missing Rate | Description |
| --- | --- |
| `η = 0.0` | Complete multimodal input |
| `0.0 < η < 1.0` | Partial missing text, audio, and vision |
| `η = 1.0` | Catastrophic simultaneous text-audio-vision missingness |

At `η = 1.0`:

- Non-special textual tokens are replaced by `[UNK]`.
- `[CLS]` and `[SEP]` are preserved.
- Audio frames are converted into zero-padded void vectors before LMMT substitution.
- Visual frames are converted into zero-padded void vectors before LMMT substitution.
- CMS-Mamba activates spatial, temporal, and numerical defense mechanisms.

CMS-Mamba does not recover missing semantics from absent modalities. Instead, it maintains a stable residual fallback state through learned structural priors, LMMT anchors, and stabilized ODE updates.

---

## 📊 Main Results

### Complete Data Performance

| Dataset | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| CMU-MOSI | 0.7496 | 0.7796 | 83.23 | 82.81 |
| CMU-MOSEI | 0.5536 | 0.7598 | 85.61 | 85.56 |

> Acc-2 and F1 are reported using the commonly used non-negative / negative-positive setting.

---

### Extreme Missingness Performance

Performance under catastrophic simultaneous text-audio-vision missingness (`η = 1.0`):

| Dataset | Has0 F1 ↑ | Non0 F1 ↑ | Mult-5 ↑ | MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| CMU-MOSI | 0.5335 | 0.5430 | 0.2128 | 1.3992 |
| CMU-MOSEI | 0.5899 | 0.4851 | 0.4136 | 0.8389 |

On CMU-MOSEI, CMS-Mamba reduces MAE from `0.9485` for the reproduced TF-Mamba baseline to `0.8389`.

---

## 📈 Average Robustness

Average performance across missing rates `η ∈ [0.0, 0.9]`:

| Dataset | Model | Mult-7 ↑ | Mult-5 ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: |
| CMU-MOSI | TF-Mamba | 31.70 | 34.54 | 1.0735 |
| CMU-MOSI | CMS-Mamba | **34.15** | **38.24** | **1.0562** |
| CMU-MOSEI | TF-Mamba | 45.53 | **46.89** | 0.6888 |
| CMU-MOSEI | CMS-Mamba | **45.68** | 46.64 | **0.6653** |

---

## 🧩 Ablation Study

Ablation results on **CMU-MOSEI**:

| Architecture | Ideal MAE ↓ | Extreme MAE ↓ | Has0 F1 ↑ | Non0 F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| TF-Mamba Baseline | 0.5560 | 0.9485 | 0.5892 | 0.4847 |
| w/o LMMT | 0.5562 | 1.0236 | 0.4785 | 0.3921 |
| w/o RNL | **0.5514** | 1.0178 | **0.5973** | **0.4926** |
| w/o DTF | 0.5603 | 0.9834 | 0.5864 | 0.4819 |
| w/ Contrastive Loss | 0.5639 | 0.9547 | 0.5878 | 0.4832 |
| CMS-Mamba Full | 0.5536 | **0.8389** | 0.5899 | 0.4851 |

Key observations:

- Removing **LMMT** weakens the classification boundary under extreme missingness.
- Removing **DTF** increases regression error under catastrophic corruption.
- Removing **RNL** causes feature-scale instability and worsens extreme MAE.
- Adding contrastive loss does not improve robustness in this setting.
- The full CMS-Mamba achieves the best extreme MAE on CMU-MOSEI.

---

## 🌏 Cross-Lingual Generalization

Results on the Chinese **CH-SIMS** dataset:

| Model | Setting | Acc-2 ↑ | Acc-3 ↑ | F1 ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-Mamba | `η = 0.0` | 73.52 | - | 74.25 | 0.4492 |
| CMS-Mamba | `η = 0.0` | **78.77** | - | **77.20** | **0.4396** |
| TF-Mamba | `η = 1.0` | 60.61 | 26.91 | - | **0.6513** |
| CMS-Mamba | `η = 1.0` | **66.96** | **30.63** | - | 0.6550 |

CMS-Mamba improves classification robustness on CH-SIMS under both complete and catastrophic missing settings.

---

## ⚡ Edge Deployment Results

CMS-Mamba is tested on **NVIDIA Jetson AGX Orin**.

Environment:

```text
Device: NVIDIA Jetson AGX Orin
L4T: R36.4.7
PyTorch: 2.5.0
CUDA: 12.6
```

| Model | Batch Size | Latency ↓ | Throughput ↑ | VRAM ↓ | Power | Temp | MAE ↓ | Has0 F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-Mamba | 16 | 90.51 ms | 176.78 samples/s | 1951.40 MB | 7.19 W | 49.96°C | 0.9482 | 0.1351 |
| TF-Mamba | 32 | OOM | OOM | OOM | - | - | - | - |
| CMS-Mamba | 16 | **79.70 ms** | **200.77 samples/s** | **648.89 MB** | 9.86 W | 52.27°C | **0.8380** | **0.5899** |
| CMS-Mamba | 32 | 106.26 ms | 300.31 samples/s | 681.17 MB | 10.15 W | 52.19°C | 0.8379 | 0.5899 |

CMS-Mamba reduces measured peak CUDA memory footprint by **66.75%** at batch size 16 compared with the reproduced TF-Mamba baseline and remains executable at batch size 32, where the baseline encounters OOM.

> The reported VRAM is the measured peak CUDA memory footprint under the tested Jetson runtime, not parameter memory alone.

---

## 🛠️ Configuration

All experiments are controlled by YAML files in `configs/`.

Example:

```yaml
dataset: mosi
model: cms_mamba
missing_rate: 0.0
batch_size: 32
learning_rate: 1e-4
epochs: 50
seed: 1111
```

---

## 📌 Recommended Workflow

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate CMSmamba

# 2. Install Mamba dependencies
pip install causal-conv1d
pip install mamba-ssm

# 3. Prepare datasets
# Put processed .pkl files under data/

# 4. Prepare offline BERT weights
# Put bert-base-uncased or bert-base-chinese in the project root

# 5. Train
python train.py --config_file configs/train_mosi.yaml

# 6. Evaluate robustness
python robust_evaluation.py --config_file configs/eval_mosi.yaml
```

---

## 📚 Citation

If this project is useful for your research, please cite:

```bibtex
@article{hu_cmsmamba,
  title={CMS-Mamba: A Context-Maintained Surviving Framework for Robust Multimodal Sentiment Analysis under Extreme Missingness},
  author={Hu, Jie and Li, Ming},
  note={Under review}
}
```

Please update the BibTeX entry after publication.

---

## 📄 License

This project is released under the MIT License.

---

## 🙏 Acknowledgements

This work was supported by the Engineering Research Center of Hubei Province for Clothing Information Program (No. 184084004) and the Hubei Key Laboratory of Digital Textile Equipment Program (No. DTL2018021).

We thank the creators of CMU-MOSI, CMU-MOSEI, CH-SIMS, Mamba, and the open-source multimodal learning community.
