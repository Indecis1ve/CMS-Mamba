# CMS-Mamba: A Context-Maintained Surviving Framework for Robust Multimodal Sentiment Analysis under Extreme Missingness

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Mamba](https://img.shields.io/badge/Backbone-Mamba-6f42c1)
![License](https://img.shields.io/badge/License-MIT-green)

Official PyTorch implementation of **CMS-Mamba**, a **Context-Maintained Surviving Mamba** framework for robust multimodal sentiment analysis under extreme modality missingness.

CMS-Mamba is designed for multimodal affective computing in uncontrolled environments, where text, audio, and visual streams may become unreliable because of token loss, sensor failure, camera occlusion, packet loss, privacy masking, feature extraction failure, or unstable edge deployment.

Instead of reconstructing absent semantics, CMS-Mamba focuses on **survivability**: when multimodal observations degrade into missing or void inputs, the model maintains a stable fallback representation through input-level anchoring, missing-aware state-space control, and final feature-scale stabilization.

---

## 🔥 Highlights

- **Extreme Missing-Modality Robustness**  
  CMS-Mamba supports robustness evaluation from missing rate `η = 0.0` to `η = 1.0`, including catastrophic simultaneous text-audio-vision missingness.

- **Spatiotemporal Orthogonal Defense**  
  CMS-Mamba introduces a hierarchical defense system:
  - **LMMT**: Learnable Missing Modality Tokens
  - **DTF**: Dynamic Time-Freezing
  - **RNL**: Representation Normalization Lock

- **State-Space Stabilization under Missingness**  
  CMS-Mamba addresses three major failure modes of Mamba-based multimodal fusion under severe degradation:
  - zero-value bias
  - state drift
  - feature-magnitude instability

- **Context-Maintained Survivability**  
  The model does not hallucinate missing affective semantics. It maintains controlled uncertainty and stable fallback behavior when observations become uninformative.

- **Long-Sequence Multimodal Modeling**  
  CMS-Mamba supports long unaligned acoustic and visual sequences, including CMU-MOSEI samples with up to 500 frames.

- **Realistic Missingness Evaluation**  
  In addition to uniform random missingness, CMS-Mamba is evaluated under block missingness, text missingness, audio-visual missingness, text-heavy corruption, audio/vision-heavy corruption, and mixed burst corruption.

- **Edge Deployment Friendly**  
  CMS-Mamba is validated on NVIDIA Jetson AGX Orin. It reduces measured peak CUDA memory footprint and remains executable at batch size 32, where the reproduced TF-Mamba baseline encounters OOM.

- **Cross-Lingual Dataset Support**  
  CMS-Mamba supports English datasets **CMU-MOSI**, **CMU-MOSEI**, and the Chinese multimodal sentiment dataset **CH-SIMS**.

---

## 🧠 Method Overview

CMS-Mamba treats robust multimodal sentiment analysis as a missing-aware state-space stabilization problem.

```text
Text / Audio / Vision Inputs
        │
        ▼
Input-Level Spatial Defense
        └── LMMT replaces missing acoustic and visual frames
            with learnable non-zero modality anchors
        │
        ▼
Text-Aware Modality Mixing
        └── Text-guided temporal alignment and modality enhancement
            for long unaligned audio/visual streams
        │
        ▼
State-Level Temporal Defense
        └── TC-Mamba with Dynamic Time-Freezing regulates
            missing-aware ODE discretization steps
        │
        ▼
Deep Query Fusion
        └── RoPE-enhanced cross-attention + TQ-Mamba
            perform sequence-level multimodal reasoning
        │
        ▼
Prediction-Level Numerical Defense
        └── RNL constrains feature-scale drift before regression
        │
        ▼
Sentiment Score Prediction
```

The core design principle is **spatial-temporal-numerical defense**:

1. **Spatial defense** prevents missing continuous modalities from collapsing into high-dimensional zero-vector manifolds.
2. **Temporal defense** prevents Mamba state updates from drifting or integrating uncontrollably under long missing sequences.
3. **Numerical defense** prevents abnormal feature magnitudes from destabilizing the final sentiment regressor.

---

## ✨ Key Components

### LMMT: Learnable Missing Modality Tokens

Conventional missing-modality pipelines often represent missing acoustic and visual frames with zero-padding. For State Space Models, repeated zero-vector inputs may introduce out-of-distribution null patterns and destabilize hidden-state evolution.

CMS-Mamba introduces **Learnable Missing Modality Tokens (LMMT)** to replace missing audio and visual frames with trainable modality-specific anchors.

For text, missing non-special tokens are replaced by `[UNK]`, while `[CLS]` and `[SEP]` are preserved to maintain valid BERT sentence boundaries.

LMMT provides stable non-zero geometric anchors, helping the model avoid zero-value bias and high-dimensional symmetry collapse.

---

### DTF: Dynamic Time-Freezing

**Dynamic Time-Freezing (DTF)** is embedded into the TC-Mamba state-space engine as a missing-aware step-size controller.

DTF regulates the effective discretization step size of the Mamba ODE system. When inputs become severely uninformative, harmful state updates are suppressed. When LMMT anchors are present, the state transition does not collapse into complete shutdown; instead, it enters a controlled positive steady state.

This allows CMS-Mamba to shift from passive suppression to stable structural absorption under catastrophic missingness.

---

### RNL: Representation Normalization Lock

Long-term processing of low-variance or missing-modality sequences can cause feature-scale drift before the final prediction layer.

CMS-Mamba applies **Representation Normalization Lock (RNL)** before the final regression head. RNL constrains the magnitude of pooled representations, suppresses numerical divergence, and stabilizes sentiment regression under extreme missingness.

---

### RoPE-Enhanced Deep Query Fusion

Under extreme missingness, modality features may become structurally stable but highly homogeneous. CMS-Mamba applies **Rotary Position Embedding (RoPE)** before cross-attention to preserve deterministic temporal geometry.

The RoPE-enhanced fused sequence is then processed by **TQ-Mamba**, enabling deep sequence-level reasoning before global pooling and final prediction.

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

For NVIDIA Jetson AGX Orin, Jetson Orin NX, or other ARM-based CUDA devices, compiling `causal-conv1d` and `mamba-ssm` from source is recommended to avoid binary compatibility issues.

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

Datasets used in this project:

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
- CMS-Mamba activates spatial, temporal, and numerical stabilization mechanisms.

The catastrophic setting `η = 1.0` is a stress test for system stability. CMS-Mamba does **not** claim to recover missing affective semantics when all modalities are absent. Instead, it maintains a stable fallback prior and avoids uncontrolled state evolution, regression explosion, and classification-boundary collapse.

---

## 📊 Main Results

### Complete Data Performance

| Dataset | MAE ↓ | Corr ↑ | Acc-2 ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: |
| CMU-MOSI | 0.7496 | 0.7796 | 83.23 | 82.81 |
| CMU-MOSEI | 0.5536 | 0.7598 | 85.61 | 85.56 |

Acc-2 and F1 are reported using the commonly used negative / non-negative or negative / positive evaluation setting.

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

CMS-Mamba achieves the best averaged MOSI metrics and lowers the averaged MOSEI MAE from `0.6888` to `0.6653`.

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

- Removing **LMMT** weakens the model under catastrophic missingness and causes classification-boundary degradation.
- Removing **DTF** increases extreme regression error, showing the importance of missing-aware temporal state control.
- Removing **RNL** may preserve ideal-state metrics but worsens extreme MAE because feature-scale drift is no longer constrained.
- Adding contrastive loss does not improve robustness in this setting.
- The full CMS-Mamba achieves the best extreme MAE on CMU-MOSEI.

---

## 🌪️ Realistic Missingness Evaluation

CMS-Mamba is further evaluated under realistic missingness patterns beyond uniform random masking:

- **Clean**: no missingness
- **Block 30% / Block 50%**: continuous temporal occlusion
- **Text Missing**: dominant textual stream is corrupted
- **A+V Missing**: audio and visual streams are missing while text remains available
- **Text-heavy**: text is severely corrupted
- **A/V-heavy**: audio and vision are severely corrupted
- **Mixed burst**: non-uniform burst-like corruption

CMS-Mamba achieves lower MAE than TF-Mamba in most realistic corruption settings, especially under clean input, block missingness, text missingness, text-heavy corruption, and audio-vision-heavy corruption.

The improvement is particularly clear when the textual stream is corrupted. Under **Text Missing**, CMS-Mamba reduces MAE from `0.9856` to `0.8453` and improves Has0 F1 from approximately `0.4440` to `0.5585`.

Under **Block 30%**, CMS-Mamba reduces MAE from `0.6340` to `0.6281` and improves correlation from `0.6537` to `0.6696`. Under **Block 50%**, it also yields lower MAE and higher correlation.

CMS-Mamba does not uniformly dominate every metric under every missingness pattern. When text remains fully available, such as in deterministic A+V Missing, TF-Mamba may obtain a slightly lower MAE, while CMS-Mamba preserves stronger correlation and classification-boundary metrics. The main advantage of CMS-Mamba lies in improving regression stability and structural robustness under severe, text-corrupted, block-corrupted, or sensor-heavy degradation.

---

## 📐 Statistical Reliability

To evaluate robustness under missing-mask sampling uncertainty, CMS-Mamba and TF-Mamba are tested on CMU-MOSEI using five missing-mask seeds:

```text
1111, 2222, 3333, 4444, 5555
```

The trained checkpoints are fixed, and only the test-time missing masks are varied.

CMS-Mamba obtains lower MAE than TF-Mamba in six out of eight realistic missingness settings. In stochastic missingness patterns, the MAE standard deviations of CMS-Mamba remain small, indicating that its robustness does not depend strongly on a particular missing-mask realization.

Notable results include:

- Clean MAE improves from `0.5561` to `0.5477`.
- Block 30% MAE improves from `0.6340` to `0.6281`.
- Block 50% MAE improves from `0.6830` to `0.6755`.
- Text Missing MAE improves from `0.9856` to `0.8453`, corresponding to a relative improvement of `14.23%`.

These results support the claim that CMS-Mamba provides reliable degradation behavior under non-uniform, structured, and asymmetric missing observations.

---

## 🌏 Cross-Lingual Generalization

CMS-Mamba is evaluated on the Chinese **CH-SIMS** dataset to test whether its defense mechanisms generalize beyond English datasets.

| Model | Setting | Acc-2 ↑ | Acc-3 ↑ | F1 ↑ | MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-Mamba | `η = 0.0` | 73.52 | - | 74.25 | 0.4492 |
| CMS-Mamba | `η = 0.0` | **78.77** | - | **77.20** | **0.4396** |
| TF-Mamba | `η = 1.0` | 60.61 | 26.91 | - | **0.6513** |
| CMS-Mamba | `η = 1.0` | **66.96** | **30.63** | - | 0.6550 |

CMS-Mamba improves classification robustness on CH-SIMS under both complete and catastrophic missing settings. The results suggest that zero-value bias, state drift, and feature-scale instability are not dataset-specific artifacts, but general vulnerabilities of multimodal State Space Models under degraded observations.

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

The reported VRAM is the measured peak CUDA memory footprint under the tested Jetson runtime, not parameter memory alone. The reduction should be interpreted as an empirical deployment-level memory advantage under the specified hardware and software setting.

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

This work was supported by the Engineering Research Center of Hubei Province for Clothing Information Program and the Hubei Key Laboratory of Digital Textile Equipment Program.

We thank the creators of CMU-MOSI, CMU-MOSEI, CH-SIMS, Mamba, TF-Mamba, and the open-source multimodal learning community.
