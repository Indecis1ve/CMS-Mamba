# CMS-Mamba

**Missing-Aware State-Space Stabilization for Robust Multimodal Sentiment Analysis under Incomplete Observations**

> **Manuscript status:** Preparing for submission. This work has not been peer-reviewed, accepted, or formally published. The code, configurations, and reported results may be updated during the submission and review process.

CMS-Mamba is a missing-aware state-space framework for robust multimodal sentiment analysis under incomplete textual, acoustic, and visual observations. It is designed to reduce unstable recurrent state updates caused by long spans of zero-valued or low-information inputs while preserving efficient Mamba-based sequence modeling.

The framework is evaluated on **CMU-MOSI**, **CMU-MOSEI**, and **CH-SIMS** under complete inputs, realistic non-uniform missingness, continuous corruption, endpoint stress testing, mechanism ablations, and NVIDIA Jetson deployment.

## Overview

Multimodal sentiment models are commonly evaluated with all modalities available. In practical systems, however, observations may become incomplete because of sensor failure, privacy masking, packet loss, occlusion, noise, or feature-extraction errors. For state-space models, missing inputs are not only absent features: they can also repeatedly affect recurrent state evolution.

CMS-Mamba addresses this problem through three stabilization levels:

* **Learnable Missing Modality Tokens (LMMT):** replace missing acoustic and visual frames with trainable non-zero anchors instead of retaining all-zero vectors.
* **Dynamic Time-Freezing (DTF):** modulate the effective Mamba discretization step according to the current representation and aligned missingness indicators.
* **Representation Normalization Lock (RNL):** apply pre-head Layer Normalization to reduce feature-scale variation before sentiment regression.

The complete architecture also uses **Text-Aware Modality Mixing (TMM)** for temporal alignment, **TC-Mamba** for context maintenance, text-guided cross-attention with auxiliary rotary positional encoding, and bidirectional **TQ-Mamba** for multimodal fusion.

## Important Interpretation of Total Missingness

The setting `eta = 1.0` is used only as a **stress-test upper bound**. At this endpoint, all mask-eligible text tokens are replaced by `[UNK]`, while the acoustic and visual streams are fully masked before CMS-Mamba applies LMMT substitution.

This setting evaluates fallback stability and controlled degradation. It does **not** imply that CMS-Mamba reconstructs or recovers semantic information that is absent from all modalities.

## Current Results

The values below are taken from the current manuscript and may be revised during submission or review.

### Complete CMU-MOSEI Test Set

Results are reported as mean ± sample standard deviation over training seeds `2024`, `2025`, and `2026`.

| Model         |               MAE ↓ |             Corr. ↑ |           Has0 F1 ↑ |
| ------------- | ------------------: | ------------------: | ------------------: |
| TF-Mamba      |     0.5562 ± 0.0018 |     0.7486 ± 0.0020 |     0.8039 ± 0.0016 |
| **CMS-Mamba** | **0.5491 ± 0.0022** | **0.7609 ± 0.0023** | **0.8083 ± 0.0013** |

### Representative CMU-MOSEI Missingness Conditions

| Condition                                  | Metric    |        TF-Mamba |           CMS-Mamba |
| ------------------------------------------ | --------- | --------------: | ------------------: |
| Block missingness, 50%                     | MAE ↓     | 0.6831 ± 0.0050 | **0.6754 ± 0.0026** |
| Text missing                               | MAE ↓     | 0.9851 ± 0.0036 | **0.8454 ± 0.0054** |
| Audio + vision missing                     | Has0 F1 ↑ | 0.7749 ± 0.0030 | **0.8334 ± 0.0036** |
| Average robustness over `eta ∈ [0.0, 0.9]` | MAE ↓     |          0.6888 |          **0.6653** |
| Simultaneous total corruption, `eta = 1.0` | MAE ↓     |          0.9485 |          **0.8389** |

Under the fixed-checkpoint endpoint stress test on CMU-MOSEI, CMS-Mamba reduces MAE from `0.9485` to `0.8389`, corresponding to an approximately **11.6% relative reduction**. Classification improvements are condition dependent and should not be interpreted as uniform across every corruption pattern.

### CH-SIMS

| Condition            | Model         |   Acc-2 ↑ |    F1 / Acc-3 ↑ |      MAE ↓ |
| -------------------- | ------------- | --------: | --------------: | ---------: |
| Complete input       | TF-Mamba      |     73.52 |        74.25 F1 |     0.4492 |
| Complete input       | **CMS-Mamba** | **78.77** |    **77.20 F1** | **0.4396** |
| Stress-test endpoint | TF-Mamba      |     60.61 |     26.91 Acc-3 | **0.6513** |
| Stress-test endpoint | **CMS-Mamba** | **66.96** | **30.63 Acc-3** |     0.6550 |

### NVIDIA Jetson AGX Orin

The following measurements use FP16 inference on Jetson AGX Orin with batch size 16. They describe one specific hardware and software configuration and should not be treated as universal deployment guarantees.

| Model         |    Latency ↓ |         Throughput ↑ |   Peak VRAM ↓ | Endpoint MAE ↓ |
| ------------- | -----------: | -------------------: | ------------: | -------------: |
| TF-Mamba      |     90.51 ms |     176.78 samples/s |    1951.40 MB |         0.9482 |
| **CMS-Mamba** | **79.70 ms** | **200.77 samples/s** | **648.89 MB** |     **0.8380** |

Jetson environment used in the manuscript: L4T R36.4.7, PyTorch 2.5.0, and CUDA 12.6.

## Project Structure

```text
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
├── data/                     # Processed dataset features
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

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Indecis1ve/CMS-Mamba.git
cd CMS-Mamba
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate CMSmamba
```

### 3. Install Mamba-SSM Dependencies

```bash
pip install causal-conv1d
pip install mamba-ssm
```

For NVIDIA Jetson or other ARM CUDA devices, compiling `causal-conv1d` and `mamba-ssm` from source is recommended for compatibility with the local CUDA, PyTorch, and system architecture.

## Data Preparation

Prepare the following datasets according to their original licenses and usage requirements:

* CMU-MOSI
* CMU-MOSEI
* CH-SIMS

Place the processed feature files in the following structure. The recommended filename is `unaligned_50.pkl`.

```text
data/
├── CMU_MOSI/
│   └── unaligned_50.pkl
├── CMU_MOSEI/
│   └── unaligned_50.pkl
└── CH_SIMS/
    └── unaligned_50.pkl
```

The current manuscript uses the following modality features:

| Dataset              | Text                     | Audio         | Vision              |
| -------------------- | ------------------------ | ------------- | ------------------- |
| CMU-MOSI / CMU-MOSEI | BERT-base, 768-D         | COVAREP, 74-D | FACET, 35-D         |
| CH-SIMS              | BERT-base-Chinese, 768-D | Librosa, 33-D | OpenFace 2.0, 709-D |

For offline BERT loading, place the pretrained weights at:

```text
./bert-base-uncased/   # English datasets
./bert-base-chinese/   # Chinese dataset
```

Complete and corrupted token sequences are encoded using the same pretrained BERT encoder. Missing non-special text tokens are replaced by `[UNK]` before BERT encoding, while `[CLS]` and `[SEP]` are preserved.

## Quick Start

### Training

```bash
# CMU-MOSI
python train.py --config_file configs/train_mosi.yaml

# CMU-MOSEI
python train.py --config_file configs/train_mosei.yaml

# CH-SIMS
python train.py --config_file configs/train_sims.yaml
```

Checkpoints are saved to:

```text
./ckpt/
```

### Robustness Evaluation

```bash
# CMU-MOSI
python robust_evaluation.py --config_file configs/eval_mosi.yaml

# CMU-MOSEI
python robust_evaluation.py --config_file configs/eval_mosei.yaml

# CH-SIMS
python robust_evaluation.py --config_file configs/eval_sims.yaml
```

Evaluation results are saved under:

```text
log/results/
```

## Experimental Protocol

The main manuscript configuration uses:

* AdamW optimization with a learning rate of `1e-4`
* `10%` warm-up followed by cosine annealing
* Weight decay of `1e-4`
* Training batch size of `64`
* Maximum of `200` epochs
* Mean squared error as the common sentiment-regression objective
* Independent training seeds `2024`, `2025`, and `2026`
* Missing-mask seeds `1111`, `2222`, `3333`, `4444`, and `5555` for stochastic test patterns
* A fixed DTF inference threshold of `tau = 0.1`

During training, a missing rate is sampled independently for each sample from `Uniform(0, 1)` and shared across text, audio, and vision. Eligible tokens or frames are then masked according to the sampled rate. Checkpoint selection uses validation MAE over multiple missing rates and fixed validation-mask seeds rather than clean-input performance alone.

Please refer to the YAML files in `configs/` for dataset-specific settings.

## Supported Corruption Patterns

The robustness evaluation covers multiple incomplete-observation settings, including:

* Independent random missingness
* Contiguous block missingness
* Complete text missingness
* Complete audio and vision missingness
* Text-heavy asymmetric missingness
* Audio/vision-heavy asymmetric missingness
* Mixed burst corruption
* Continuous simultaneous corruption from `eta = 0.0` to `eta = 1.0`

## Reproducibility Notes

* Use the same processed feature files for model comparison.
* Keep training seeds and missing-mask seeds fixed when reproducing manuscript tables.
* Do not select checkpoints or classification thresholds using the test set.
* Server-side experiments in the manuscript use FP32 on a single NVIDIA A30 GPU.
* Jetson measurements use FP16 and may vary with software versions, kernels, compilation flags, power mode, and thermal conditions.
* The endpoint `eta = 1.0` measures learned fallback stability, not missing-semantic reconstruction.

## Checkpoints

Place trained or downloaded checkpoints in `ckpt/`. Checkpoint filenames and paths should match the corresponding evaluation YAML configuration.

Public checkpoint availability may be updated as the repository is prepared for submission.

## Citation

The manuscript is currently being prepared for submission. Until a public preprint or final bibliographic record is available, the following provisional citation may be used:

```bibtex
@unpublished{hu2026cmsmamba,
  title  = {CMS-Mamba: Missing-Aware State-Space Stabilization for Robust Multimodal Sentiment Analysis under Incomplete Observations},
  author = {Hu, Jie and Dang, Qingxia and Li, Ming},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

Please update the citation after a public preprint or final publication record becomes available.

## Acknowledgments

This work was supported by the Engineering Research Center of Hubei Province for Clothing Information Program under Grant No. 184084004.

## Contact

For questions about the method or implementation, contact:

* **Ming Li**: `lettermail@wtu.edu.cn`

## Disclaimer

This repository is research code accompanying a manuscript in preparation. Results may differ across hardware, software environments, dataset preprocessing pipelines, random seeds, and checkpoint-selection procedures. Users are responsible for complying with the licenses and terms of the original datasets, pretrained models, and third-party dependencies.
