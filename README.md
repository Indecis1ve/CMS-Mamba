# CMS-Mamba

PyTorch implementation accompanying **Robust Multimodal Sentiment Analysis under Incomplete Observations via Missingness-Conditioned State-Space Modulation**.

**Authors:** Jie Hu, Qingxia Dang, and Ming Li  
**Manuscript status:** Under review  
**Repository:** [Indecis1ve/CMS-Mamba](https://github.com/Indecis1ve/CMS-Mamba)

CMS-Mamba studies how recurrent states should evolve when textual, acoustic, or visual observations are incomplete. Its central mechanism, **Missingness-Conditioned Selective State-Space Modulation (MCSSM)**, conditions the effective discretization step of selective state-space layers on aligned features and missingness. The deployable **CMS-Mamba-Auto** variant estimates missingness from degraded inputs during both training and inference.

The revised manuscript emphasizes controlled mechanism validation: MCSSM improves three matched Mamba-family backbones, while the complete framework combines MCSSM with supporting input substitution and pre-head normalization. The main contribution is missingness-conditioned state integration, rather than a claim of universal state-of-the-art performance.

## Model variants and terminology

| Name | Meaning | MCSSM conditioning |
| --- | --- | --- |
| CMS-Mamba-Auto | Complete deployable framework | Input-derived indicators during training and inference |
| CMS-Mamba (Ref.) | Reference-conditioned mechanism control | Corruption-process indicators during training and evaluation |
| Backbone + MCSSM-Auto | Cross-backbone mechanism-isolation variant | Automatic indicators; only MCSSM is added |

The cross-backbone variants do **not** add LMMT or additional pre-head normalization. Their results must not be confused with those of the complete CMS-Mamba-Auto framework.

Earlier documentation used **DTF** for state-update control and **RNL** for prediction-stage normalization. This README follows the revised manuscript: **MCSSM** is the state-space contribution, **LMMT** is a supporting learned-placeholder design, and **pre-head LayerNorm** is a standard normalization component.

## Installation

The documented repository entry points use a Conda environment and a CUDA-enabled PyTorch installation:

```bash
git clone https://github.com/Indecis1ve/CMS-Mamba.git
cd CMS-Mamba
conda env create -f environment.yml
conda activate CMSmamba
```

The reference software stack reported in the revised manuscript is:

| Component | Reported version |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.5.0 |
| CUDA | 12.6 |
| cuDNN | 9.3.0 |
| Transformers | 4.46.3 |
| mamba-ssm | 2.2.2 |

Use CUDA extensions compatible with the selected PyTorch/CUDA stack, including the selective-scan and causal-convolution dependencies. Jetson/ARM installations require platform-compatible builds. The software stack above describes the manuscript experiments; it does not imply that a server environment can be copied unchanged to Jetson.

## Data preparation

The sentiment experiments use the official TF-Mamba feature release and the corresponding dataset splits.

| Dataset | Text encoder | Acoustic features | Visual features | Maximum aligned length |
| --- | --- | --- | --- | --- |
| CMU-MOSI | BERT-base, 768 dimensions | COVAREP, 74 dimensions | FACET, 35 dimensions | 50 |
| CMU-MOSEI | BERT-base, 768 dimensions | COVAREP, 74 dimensions | FACET, 35 dimensions | 50 |
| CH-SIMS | BERT-base-Chinese, 768 dimensions | Librosa, 33 dimensions | OpenFace 2.0, 709 dimensions | 39 |

The documented data and pretrained-encoder layout is shown below. Adjust the dataset and encoder paths in the YAML configuration files to match your installation:

```text
data/CMU_MOSI/unaligned_50.pkl
data/CMU_MOSEI/unaligned_50.pkl
data/CH_SIMS/unaligned_50.pkl
bert-base-uncased/
bert-base-chinese/
```

The processed-file name does not define the model's aligned sequence length; use the dataset-specific configuration. Keep the official train/validation/test split separation when preparing features and calibrating the missingness estimator.

## Training and evaluation

The commands below retain the repository's documented MOSI, MOSEI, and CH-SIMS entry points.

### Training

```bash
python train.py --config_file configs/train_mosi.yaml
python train.py --config_file configs/train_mosei.yaml
python train.py --config_file configs/train_sims.yaml
```

### Robustness evaluation

```bash
python robust_evaluation.py --config_file configs/eval_mosi.yaml
python robust_evaluation.py --config_file configs/eval_mosei.yaml
python robust_evaluation.py --config_file configs/eval_sims.yaml
```

The documented output locations are `ckpt/` for checkpoints and `log/results/` for evaluation results. Configure data, pretrained-encoder, and checkpoint paths before running the scripts. For manuscript comparisons, distinguish the Auto and Ref. protocols defined above and use matching configurations and checkpoints.

## Method

### MCSSM: control of state-space integration

For an aligned representation $x_t$ and its branch-specific missingness vector $m_t$, MCSSM computes

$$
\alpha_t = \sigma(W_g x_t + W_m m_t + b_g),
\qquad
\Delta_{\mathrm{base},t} = \operatorname{Softplus}(W_\Delta x_t + b_\Delta),
\qquad
\Delta_t = \alpha_t \Delta_{\mathrm{base},t}.
$$

The effective step controls both terms of the zero-order-hold state update:

$$
\bar A_t = \exp(\Delta_t A),
\qquad
\bar B_t = \int_0^{\Delta_t}\exp((\Delta_t-\tau)A)B\,d\tau,
\qquad
h_t = \bar A_t h_{t-1} + \bar B_t x_t.
$$

As the effective step approaches zero, the transition approaches the identity and input injection approaches zero. A positive step can instead integrate a learned fallback representation. MCSSM therefore regulates recurrent integration, rather than simply dropping features or multiplying the output of an already completed recurrent update.

The integration gate $\alpha_t$ is not a calibrated missingness probability. The same continuous integration rule is used in training and inference; no separate hard-freeze threshold is introduced.

### Supporting components

- **LMMT:** two jointly learned modality-specific vectors replace detected missing acoustic and visual frames before alignment. They provide stable fallback representations, not reconstruction of unavailable semantic content.
- **Inherited backbone:** text-aware alignment, multimodal fusion, and the text-reconstruction branch and objective are retained from TF-Mamba.
- **Pre-head LayerNorm:** standard normalization limits pooled-feature scale variation before regression.

The manuscript reports all eight LMMT/MCSSM/pre-head-normalization combinations, with RoPE fixed, and a separate RoPE ablation. Complete-framework gains are not attributed to MCSSM alone.

### Automatic missingness estimation

For continuous modality $m$, the score is computed before substitution and alignment:

$$
q_{m,t} = \frac{\lVert\widetilde{x}_{m,t}\rVert_2}{\sqrt{d_m}}.
$$

Exactly zero or non-finite frames are marked missing directly. A nonzero low-energy frame is marked missing when its score and at least one valid temporal neighbor's score are below the modality-specific threshold. Text uses non-special `[UNK]` token positions; `[CLS]` and `[SEP]` remain observed, and padding is excluded.

For each dataset-seed run, feature standardization is fitted on the training split and thresholds are calibrated on the validation split by missing-class F1. Test data and sentiment labels are not used to fit these preprocessing rules. The nominal CMU-MOSEI thresholds are **0.18 for audio** and **0.21 for vision**; these are not universal thresholds for arbitrary feature extractors.

The processing order is:

1. Standardize received continuous features using training-split statistics.
2. Estimate raw-frame and token missingness, excluding padding and respecting valid temporal boundaries.
3. Apply LMMT substitution to detected missing continuous frames; represent text corruption through `[UNK]` substitution before BERT encoding.
4. Align features and propagate the selected MCSSM indicators through the same text-centric alignment matrices.
5. Construct visual-text and acoustic-text missingness vectors and run MCSSM in both TC-Mamba branches.
6. Apply the inherited fusion pathway, pooling, pre-head normalization, and prediction head.

The indicator thresholding operation is discrete. Training gradients propagate through the selected learned tokens, alignment, MCSSM, and downstream modules, but not through the heuristic threshold decisions.

In reference-conditioned controls, only the MCSSM conditioning signal is replaced with corruption-process indicators. LMMT retains its input-derived substitution decision.

## Experimental protocol

- **Training seeds:** 2024, 2025, 2026, 2027, and 2028 for the main five-seed experiments.
- **Validation mask seeds:** 1111, 2222, and 3333.
- **Test mask seeds:** 1111, 2222, 3333, 4444, and 5555 for stochastic patterns.
- **Checkpoint selection:** lowest validation MAE over missing rates 0.0, 0.1, 0.3, 0.5, and 0.7 and the three validation mask seeds; all metrics use the same selected checkpoint.
- **Seven-pattern incomplete macro:** Block 30%, Block 50%, Text Missing, A+V Missing, Text-heavy, A/V-heavy, and Mixed burst. Clean input is excluded.
- **Continuous-rate summary:** the evaluated grid within $\eta\in[0.0,0.9]$, excluding the total-missingness endpoint.
- **Uncertainty:** mean and sample standard deviation across training seeds. Stochastic realizations are averaged within each seed before computing the macro score and across-seed SD.
- **Statistical comparisons:** paired differences and 95% confidence intervals use matched seeds; Holm correction is applied separately to the explicitly defined comparison families.

The TF-Mamba/CMS-Mamba training setup uses AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$, batch size 64, up to 200 epochs, and 10% warm-up followed by cosine annealing. The inherited reconstruction-loss weights are 0.7 for MOSI, 0.3 for MOSEI, and 1.0 for CH-SIMS. Full configuration details and comparison controls are given in the manuscript and supplementary material.

At $\eta=1.0$, all mask-eligible observations are removed, while boundary-token structure and learned continuous-modality fallback representations remain. Endpoint performance characterizes fallback behavior under this protocol, not recovery of the removed semantic content.

## Results from the revised manuscript

The following results are from the revised manuscript and supplementary material. Unless stated otherwise, uncertainty is the sample SD across five training seeds. Lower MAE is better.

### Deployable CMU-MOSEI comparison

Selected matched-protocol results for the seven-pattern incomplete-input macro MAE:

| Model | Macro MAE, mean ± sample SD |
| --- | --- |
| TF-Mamba | 0.6979 ± 0.0066 |
| Retrained no-indicator model | 0.6846 ± 0.0057 |
| **CMS-Mamba-Auto** | **0.6775 ± 0.0015** |
| LNLN | 0.6737 ± 0.0060 |
| APRD | 0.6742 ± 0.0049 |
| MSLMamba | 0.6748 ± 0.0054 |
| EASE | 0.6755 ± 0.0053 |
| CTRN | 0.6768 ± 0.0039 |

CMS-Mamba-Auto reduces MAE relative to TF-Mamba by **0.0204**, with paired 95% CI **[0.0130, 0.0278]** and two-test Holm **p = 0.0031**. Its complete-input MAE is **0.5491 ± 0.0018**.

The five recent baselines listed below CMS-Mamba-Auto have lower macro-MAE point estimates, but their paired differences from Auto are not significant after the separate five-test Holm correction. This does not establish equivalence; the contribution is evaluated through matched-backbone and mechanism controls.

The reference-conditioned CMS-Mamba control reaches **0.6731 ± 0.0022** macro MAE. It uses reference indicators for MCSSM and is not part of the deployable-model ranking.

For Auto versus the retrained no-indicator model, the prespecified five-seed comparison has Holm **p = 0.0740**. The separately extended eight-seed pairing gives the same **0.0071** reduction, paired 95% CI **[0.0027, 0.0115]**, and **unadjusted p = 0.0067**. These two analyses retain their separate statistical interpretations.

### Cross-backbone portability: only MCSSM-Auto added

No LMMT or additional pre-head normalization is added in these comparisons. Training settings are matched within each backbone, and automatic indicators are used during both training and evaluation.

| Backbone | Original macro MAE | + MCSSM-Auto macro MAE | Paired reduction [95% CI] | Holm p |
| --- | --- | --- | --- | --- |
| TF-Mamba | 0.6979 ± 0.0066 | 0.6863 ± 0.0048 | 0.0116 [0.0069, 0.0163] | 0.0072 |
| MSLMamba | 0.6748 ± 0.0054 | 0.6671 ± 0.0046 | 0.0077 [0.0021, 0.0133] | 0.0186 |
| Vanilla early-fusion Mamba | 0.7162 ± 0.0029 | 0.7029 ± 0.0058 | 0.0133 [0.0076, 0.0190] | 0.0072 |

All three paired reductions remain significant after correction across the three backbone comparisons. Confidence intervals are unadjusted paired 95% intervals; the p values are Holm-adjusted.

### Mechanism, detector, and fallback controls

- **Conditioning location:** with the other components fixed and reference indicators supplied, MCSSM significantly improves on binary step freezing and input-feature gating under both Text Missing and total missingness. It also improves on output-representation gating at total missingness. These are five significant contrasts in the shared 16-test Holm family. Differences from additive-step, indicator-only, and feature-only step modulation are not resolved after correction.
- **Mean rankings:** MCSSM has the lowest mean MAE for Block 50%, Text Missing, Mixed burst, and total missingness in the location comparison. Under A+V Missing, the hidden-state gate is slightly lower: 0.5747 versus 0.5749.
- **Matched dropout controls:** CMS-Mamba-Auto reduces macro MAE by 0.0138 versus feature dropout and 0.0087 versus modality dropout; both two-test Holm p values are 0.0003. These are complete-framework comparisons, distinct from the MCSSM-only portability experiment.
- **Learned detector:** a lightweight MLP improves missingness-detection macro F1 from 0.934 to 0.948, but increases downstream macro MAE from 0.6775 to 0.6783. Learned-minus-heuristic MAE is +0.0008, paired 95% CI [0.0003, 0.0013], p = 0.0155. The heuristic is therefore retained for the reported end-to-end pipeline.
- **LMMT alternatives:** LMMT improves over zero vectors, train-split means, and training-set k-means prototypes under both Text Missing and total missingness; all six Holm-adjusted p values are at most 0.0147. Training-set prototypes are distinct from representations supplied by external pretrained modality encoders.

### Distribution, dataset, and task coverage

- **OOD perturbations:** without retraining for each listed perturbation, Auto has lower mean MAE in 11 of 12 feature/text-space conditions, with a mean reduction of 0.0180 across all 12. Mild FACET temporal smoothing is the exception, with Auto higher by 0.0004. The same 11 improved conditions also have smaller across-seed MAE SDs.
- **CMU-MOSI continuous missingness:** the reference-conditioned model lowers trajectory-average MAE from 1.0735 ± 0.0163 to 1.0562 ± 0.0127 and improves Mult-7/Mult-5 by 2.45/3.70 percentage points. Its Has0 F1 is higher at the total-missingness endpoint, but does not consistently exceed TF-Mamba at intermediate missing rates.
- **CH-SIMS:** Auto lowers complete-input MAE from 0.4492 to 0.4421 and improves Acc-2 from 73.52% to 77.84%. At total missingness, Acc-2 improves from 60.61% to 65.18%, while MAE increases from 0.6513 to 0.6584. All four corresponding paired differences survive the four-test Holm correction. This is cross-dataset applicability with dataset-specific training, not zero-shot cross-lingual transfer.
- **IEMOCAP emotion recognition:** the session-independent four-class experiment extends evaluation beyond sentiment regression. Incomplete-macro WF1 increases from 63.17 ± 0.88 to 65.28 ± 0.81; the paired gain is 2.11 points, 95% CI [0.88, 3.34], p = 0.0090.

The extended experiments are specified in the revised manuscript and supplementary material; the quick-start commands above cover the existing documented sentiment-dataset entry points.

## Computational and deployment measurements

The complete CMS-Mamba-Auto framework has **3.31M parameters and 0.89G MACs**, compared with **3.24M and 0.86G** for TF-Mamba.

| Model / inference path | A30 FP32 latency (ms/sample) | Jetson FP16 latency (ms/B16) | Jetson energy (J/sample) |
| --- | --- | --- | --- |
| TF-Mamba | 2.84 ± 0.09 | 90.51 ± 1.84 | 0.0407 |
| CMS-Mamba-Auto Core | 2.71 ± 0.08 | 79.70 ± 1.53 | 0.0491 |
| CMS-Mamba-Auto E2E | 2.75 ± 0.08 | 80.82 ± 1.61 | 0.0501 |

Timing uncertainty is mean ± sample SD over 30 synchronized runs after warm-up. Jetson measurements use AGX Orin 32 GB, FP16, MAXN mode, locked clocks, and active cooling.

- **Core** uses precomputed automatic indicators and excludes detector preprocessing.
- **E2E** includes automatic missingness estimation and associated preprocessing, LMMT substitution, alignment, state-space processing, and prediction. The A30 E2E value, **2.75 ± 0.08 ms/sample**, is directly measured.
- These are **feature-level** end-to-end measurements: upstream raw audio/video feature extraction and text encoding are outside the timed region.
- Across Clean, Block 50%, Text Missing, and A+V Missing on Jetson at batch size 16, preprocessing accounts for **1.30–1.61% of total E2E latency**. The denominator is E2E latency, not Core latency.
- The no-indicator runtime row in the supplementary material is a separately retrained graph, not a subtraction-based estimate of detector cost.

CMS-Mamba-Auto is faster under the reported runtime, but consumes more recorded energy per sample than TF-Mamba. Parameter counts, MACs, latency, and energy describe different properties; the results support low added complexity and lower measured latency, not a general energy-efficiency advantage.

## Scope and open directions

Current evidence covers three Mamba-family backbones, controlled missingness, feature/text-space OOD perturbations, sentiment analysis, and preliminary emotion classification. Future work includes transfer across feature extractors and model families, task-aware learned missingness detection, external pretrained fallback representations, and joint optimization of robustness, latency, and energy under raw-sensor deployment.

## Citation

The manuscript is under review. Please use the current title and complete author list:

```bibtex
@misc{hu2026cmsmamba,
  title  = {Robust Multimodal Sentiment Analysis under Incomplete Observations via Missingness-Conditioned State-Space Modulation},
  author = {Hu, Jie and Dang, Qingxia and Li, Ming},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/Indecis1ve/CMS-Mamba}
}
```

## License

This project is released under the MIT License. Third-party datasets, pretrained models, and dependencies remain subject to their respective licenses.

## Acknowledgments

This work was supported by the Engineering Research Center of Hubei Province for Clothing Information Program (No. 184084004).

We thank the creators and maintainers of CMU-MOSI, CMU-MOSEI, CH-SIMS, IEMOCAP, Mamba, TF-Mamba, and the open-source multimodal learning community.
