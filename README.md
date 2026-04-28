# CMS-Mamba: A Context-Maintained Surviving Framework for Robust Multimodal Sentiment Analysis under Extreme Missingness

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Mamba](https://img.shields.io/badge/Backbone-Mamba-6f42c1)
![License](https://img.shields.io/badge/License-MIT-green)

Official PyTorch implementation of **CMS-Mamba**, a **Context-Maintained Surviving Mamba** framework for robust multimodal sentiment analysis under extreme modality missingness.

CMS-Mamba is designed for real-world multimodal affective computing scenarios where text, audio, and visual signals may be partially or even completely corrupted due to sensor failure, camera occlusion, packet loss, privacy masking, or unstable edge-device deployment.

Unlike conventional missing-modality methods that rely on reconstruction or zero-padding, CMS-Mamba introduces a hierarchical defense system to stabilize State Space Model dynamics under severe data degradation.

---

## 🔥 Highlights

- **Extreme Missing-Modality Robustness**  
  Supports robustness evaluation from missing rate `η = 0.0` to `η = 1.0`, including the catastrophic case where text, audio, and vision are simultaneously corrupted.

- **Spatiotemporal Orthogonal Defense**  
  A three-layer defense framework consisting of:
  - **LMMT**: Learnable Missing Modality Tokens
  - **DTF**: Dynamic Time-Freezing
  - **RNL**: Representation Normalization Lock

- **State-Space Stability under Missingness**  
  CMS-Mamba directly addresses zero-value bias, hidden-state degradation, and integral explosion in Mamba-based multimodal fusion.

- **Long-Sequence Multimodal Modeling**  
  Efficiently handles long acoustic and visual streams, especially on CMU-MOSEI where sequences can reach up to 500 frames.

- **Edge Deployment Friendly**  
  Verified on NVIDIA Jetson AGX Orin, CMS-Mamba significantly reduces memory footprint and avoids Out-Of-Memory failures compared with the baseline TF-Mamba.

- **Multilingual Dataset Support**  
  Compatible with English datasets such as **CMU-MOSI** and **CMU-MOSEI**, as well as the Chinese multimodal sentiment dataset **CH-SIMS**.

---

## 🧠 Method Overview

CMS-Mamba is built upon a text-guided Mamba fusion architecture and enhances it with a hierarchical survival mechanism.

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
 └── CTC-based temporal alignment
        │
        ▼
Temporal Defense
 └── TC-Mamba with Dynamic Time-Freezing
        │
        ▼
Deep Query Fusion
 └── TQ-Mamba with text-guided query reasoning
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

### 1. LMMT: Learnable Missing Modality Tokens

**Learnable Missing Modality Tokens** replace zero-padded audio and visual frames with trainable geometric anchors.

Instead of feeding pure zero vectors into the model, LMMT provides stable non-zero representations for missing continuous modalities, preventing the hidden feature manifold from collapsing.

For text, missing tokens are replaced by `[UNK]`, while `[CLS]` and `[SEP]` are preserved to maintain valid BERT sentence boundaries.

---

### 2. DTF: Dynamic Time-Freezing

**Dynamic Time-Freezing** is embedded into the Mamba state-space engine.

When missing or highly degraded frames are detected, DTF dynamically adjusts the discretization step size of the Mamba ODE system, forcing the state transition matrix to approach an identity mapping.

As a result, historical emotional context is preserved rather than being overwritten by corrupted inputs.

---

### 3. RNL: Representation Normalization Lock

**Representation Normalization Lock** is applied before the final regression head.

It suppresses numerical divergence and feature magnitude explosion caused by long-term integration of missing-modality representations, improving regression stability under extreme missingness.

---

## 📁 Directory Structure

```text
CMSmamba/
├── ckpt/
├── configs/                      # YAML configuration files for training and evaluation
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
├── data/                       # Dataset directory, manually prepared
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

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/CMS-Mamba.git
cd CMS-Mamba
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate CMSmamba
```

### 3. Install Mamba Dependencies

CMS-Mamba relies on `causal-conv1d` and `mamba-ssm`.

```bash
pip install causal-conv1d
pip install mamba-ssm
```

> **Note**
>
> If you deploy CMS-Mamba on edge platforms such as NVIDIA Jetson AGX Orin, Jetson Orin NX, or other ARM-based CUDA devices, it is recommended to compile `causal-conv1d` and `mamba-ssm` from source to avoid binary compatibility issues.

---

## 📦 Data Preparation

Please manually download and preprocess the multimodal feature files for each dataset.

The processed `.pkl` files should be placed in the corresponding dataset directories.

Example:

```text
data/CMU_MOSI/Processed/unaligned_50.pkl
data/CMU_MOSEI/Processed/unaligned_50.pkl
data/CH_SIMS/Processed/unaligned_50.pkl
```

Recommended data organization:

```text
data/
├── CMU_MOSI/
│       └── unaligned_50.pkl
│
├── CMU_MOSEI/
│       └── unaligned_50.pkl
│
└── CH_SIMS/
        └── unaligned_50.pkl
```

---

## 🤗 Offline BERT Weights

To support offline HPC clusters and edge devices without internet access, CMS-Mamba uses local BERT loading.

Please download the required HuggingFace BERT weights and place them in the project root directory.

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
│
└── bert-base-chinese/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── vocab.txt
```

---

## 🚀 Quick Start

### 1. Training on CMU-MOSI

```bash
python train.py --config_file configs/train_mosi.yaml
```

The best checkpoint will be saved to:

```text
ckpt/mosi/
```

---

### 2. Training on CMU-MOSEI

```bash
python train.py --config_file configs/train_mosei.yaml
```

The best checkpoint will be saved to:

```text
ckpt/mosei/
```

---

### 3. Training on CH-SIMS

```bash
python train.py --config_file configs/train_sims.yaml
```

The best checkpoint will be saved to:

```text
ckpt/sims/
```

---

## 🧪 Robustness Evaluation

CMS-Mamba supports robustness evaluation across a full missing-rate spectrum from `η = 0.0` to `η = 1.0`.

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

The evaluation results will be saved in:

```text
log/results/
```


---

## 🔬 Missing-Modality Protocol

During robustness evaluation, CMS-Mamba simulates missingness only on the test set.

The missing rate is denoted as `η`.

| Missing Rate | Description |
| --- | --- |
| `η = 0.0` | Complete multimodal input |
| `0.0 < η < 1.0` | Partial missing text, audio, and vision |
| `η = 1.0` | Catastrophic simultaneous missingness |

At `η = 1.0`:

- Non-special textual tokens are replaced by `[UNK]`.
- Audio frames are replaced by zero-padded void vectors.
- Visual frames are replaced by zero-padded void vectors.
- CMS-Mamba activates its spatial, temporal, and numerical defense mechanisms to prevent representation collapse.

---

## 📊 Main Results

### Complete Data Performance

CMS-Mamba remains competitive under the ideal complete-data setting.

| Dataset | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| CMU-MOSI | 0.7496 | 0.7796 | 83.23 | 82.81 |
| CMU-MOSEI | 0.5536 | 0.7598 | 85.61 | 85.56 |

> The Acc-2 and F1 values follow the commonly used non-negative / negative-positive setting reported in multimodal sentiment analysis.

---

### Extreme Missingness Performance

Under catastrophic simultaneous text-audio-vision missingness, CMS-Mamba maintains stable regression and classification behavior.

| Dataset | Missing Rate | Has0 F1 ↑ | Non0 F1 ↑ | Mult-5 ↑ | MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMU-MOSI | `η = 1.0` | 0.5335 | 0.5430 | 0.2128 | 1.3992 |
| CMU-MOSEI | `η = 1.0` | 0.5899 | 0.4851 | 0.4136 | 0.8389 |

On CMU-MOSEI, the baseline TF-Mamba suffers severe degradation under `η = 1.0`, with MAE increasing to `0.9485`, while CMS-Mamba suppresses the error to `0.8389`.

---

## 📈 Average Robustness

Average robustness is measured across multiple missing rates from `η = 0.0` to `η = 0.9`.

| Dataset | Model | Mult-7 ↑ | Mult-5 ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: |
| CMU-MOSI | TF-Mamba | 31.70 | 34.54 | 1.0735 |
| CMU-MOSI | CMS-Mamba | **34.15** | **38.24** | **1.0562** |
| CMU-MOSEI | TF-Mamba | 45.53 | **46.89** | 0.6888 |
| CMU-MOSEI | CMS-Mamba | **45.68** | 46.64 | **0.6653** |

---

## 🧩 Ablation Study

The ablation study verifies the contribution of each defense component.

| Architecture | Ideal MAE ↓ | Extreme MAE ↓ | Has0 F1 ↑ | Non0 F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| TF-Mamba Baseline | 0.7884 | **1.3736** | 0.3932 | 0.4231 |
| w/o LMMT | 0.7742 | 1.4327 | 0.2767 | 0.2507 |
| w/o RNL | **0.7470** | 1.5169 | 0.5097 | 0.5214 |
| w/o DTF | 0.7970 | 1.5062 | 0.3932 | 0.4231 |
| w/ Contrastive Loss | 0.8028 | 1.3857 | 0.3932 | 0.4231 |
| CMS-Mamba Full | 0.7496 | 1.3992 | **0.5335** | **0.5430** |

Key observations:

- Removing **LMMT** causes severe classification boundary collapse.
- Removing **DTF** leads to temporal integration instability.
- Removing **RNL** causes regression error explosion under extreme missingness.
- Adding contrastive loss does not improve robustness and may cause feature homogenization.
- The full CMS-Mamba achieves the best classification survivability under catastrophic missingness.

---

## 🌏 Cross-Lingual Generalization

CMS-Mamba is also evaluated on the Chinese CH-SIMS dataset.

| Model | Setting | Acc-2 ↑ | Acc-3 ↑ | F1 ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-Mamba | `η = 0.0` | 73.52 | - | 74.25 | 0.4492 |
| CMS-Mamba | `η = 0.0` | **78.77** | - | **77.20** | **0.4396** |
| TF-Mamba | `η = 1.0` | 60.61 | 26.91 | - | **0.6513** |
| CMS-Mamba | `η = 1.0` | **66.96** | **30.63** | - | 0.6550 |

These results show that CMS-Mamba provides language-agnostic robustness against zero-value bias and state-space instability.

---

## ⚡ Edge Deployment Results

CMS-Mamba is tested on an NVIDIA Jetson AGX Orin edge platform.

Environment:

```text
Device: NVIDIA Jetson AGX Orin
L4T: R36.4.7
PyTorch: 2.5.0
CUDA: 12.6
```

| Model | Batch Size | Latency ↓ | Throughput ↑ | VRAM ↓ | MAE ↓ | Has0 F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-Mamba | 16 | 90.51 ms | 176.78 samples/s | 1951.40 MB | 0.9482 | 0.1351 |
| TF-Mamba | 32 | OOM | OOM | OOM | - | - |
| CMS-Mamba | 16 | **79.70 ms** | **200.77 samples/s** | **648.89 MB** | **0.8380** | **0.5899** |
| CMS-Mamba | 32 | 106.26 ms | 300.31 samples/s | 681.17 MB | 0.8379 | 0.5899 |

CMS-Mamba reduces VRAM usage by nearly **66%** at batch size 16 and avoids the OOM failure encountered by the baseline at batch size 32.

---

## 🛠️ Configuration

All experiments are controlled by YAML configuration files in the `configs/` directory.

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

# 2. Prepare datasets
# Put processed .pkl files into the data directory

# 3. Prepare offline BERT weights
# Put bert-base-uncased or bert-base-chinese in the project root

# 4. Train model
python train.py --config_file configs/train_mosi.yaml

# 5. Evaluate robustness
python robust_evaluation.py --config_file configs/eval_mosi.yaml
```

---

## 📚 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{cmsmamba2024,
  title={CMS-Mamba: A Context-Maintained Surviving Framework for Robust Multimodal Sentiment Analysis under Extreme Missingness},
  author={Jie Hu and Ming Li},
  journal={Knowledge-Based Systems},
  year={2024},
  note={Under Review}
}
```

---

## 📄 License

This project is released under the MIT License.

---

## 🙏 Acknowledgements

This project builds upon the development of multimodal sentiment analysis, State Space Models, Mamba, and robust missing-modality learning.

We sincerely thank the creators of the CMU-MOSI, CMU-MOSEI, CH-SIMS datasets and the open-source research community.
