# CMS-Mamba

**Missing-Aware State-Space Stabilization for Robust Multimodal Sentiment Analysis under Incomplete Observations**

CMS-Mamba is the reference implementation accompanying a manuscript in preparation. It targets sentiment regression from incomplete text, audio, and vision streams on CMU-MOSI, CMU-MOSEI, and CH-SIMS.

> Manuscript status: not peer reviewed or formally published. The implementation and experimental protocol may change during review.

## Paper-aligned implementation

The current code follows the manuscript's method:

- explicit validity masks and missingness masks (`1 = missing`);
- one shared per-sample training rate `eta ~ Uniform(0, 1)` with independent Bernoulli masking in text, audio, and vision;
- `[UNK]` replacement for missing non-special text tokens, preserving `[CLS]`, `[SEP]`, and padding;
- Learnable Missing Modality Tokens (LMMT) for missing audio and visual frames;
- Text-Aware Modality Mixing (TMM), including alignment of missingness indicators with the same alignment probabilities;
- branch-specific, bidirectional TC-Mamba with Dynamic Time-Freezing (DTF);
- separate audio-text and vision-text contexts until the fusion boundary;
- feature-wise audio/vision concatenation, text-guided cross-attention, and key-only RoPE;
- bidirectional TQ-Mamba, adaptive max pooling, Representation Normalization Lock (RNL), and a linear regression head;
- MSE-only optimization;
- checkpoint selection by mean validation MAE over a fixed missingness grid.

DTF uses a continuous sigmoid gate during training. Evaluation applies the manuscript threshold `tau = 0.1`. Total missingness is a fallback-stability stress test, not semantic reconstruction.

## Compatibility notice

This revision changes parameter names, shapes, and the forward data contract. Checkpoints produced by the previous implementation are intentionally incompatible. `robust_evaluation.py` loads weights with `strict=True` and requests retraining when an old checkpoint is supplied.

No model training, dataset evaluation, benchmark, or performance simulation was run as part of this code-alignment revision. The included tests use only tiny deterministic arrays and tensors; they do not reproduce manuscript results.

## Repository layout

```text
CMS-Mamba/
|-- configs/                 # train/evaluation configurations
|-- core/
|   |-- dataset.py           # data contract and deterministic corruption
|   |-- missingness.py       # training/evaluation missingness policies
|   |-- validation.py        # validation grid and checkpoint selector
|   `-- losses.py            # MSE-only objective
|-- models/
|   |-- missingness.py       # LMMT and DTF
|   |-- tmm.py               # alignment and mask propagation
|   |-- mamba.py             # TC/TQ stacks and fusion
|   |-- TFMamba.py           # complete CMS-Mamba model
|   `-- mamba_nets/          # selective-scan implementation
|-- tests/                   # lightweight unit and contract tests
|-- train.py
`-- robust_evaluation.py
```

## Environment

The supplied environment targets Python 3.10, PyTorch 2.1, CUDA 12.1, Transformers, and Mamba-SSM.

```bash
conda env create -n CMSmamba -f environment.yml
conda activate CMSmamba
```

For CUDA or NVIDIA Jetson systems, `causal-conv1d` and `mamba-ssm` may need to be compiled against the installed PyTorch/CUDA combination.

## Data contract

Prepare the datasets according to their original licenses. Each pickle must contain `train`, `valid`, and `test` splits with at least:

- `text_bert`: `[N, 3, L_text]` input IDs, attention masks, and token-type IDs;
- `audio`: `[N, L_audio, D_audio]`;
- `vision`: `[N, L_vision, D_vision]`;
- `audio_lengths` and `vision_lengths`;
- the configured regression-label field.

The manuscript feature dimensions are:

| Dataset | Text | Vision | Audio |
| --- | ---: | ---: | ---: |
| CMU-MOSI / CMU-MOSEI | BERT-base, 768 | FACET, 35 | COVAREP, 74 |
| CH-SIMS | BERT-base-Chinese, 768 | OpenFace 2.0, 709 | Librosa, 33 |

Configured dimensions are validated when the dataset is loaded. A mismatched preprocessing pipeline fails with an actionable error.

Place offline pretrained encoders at `./bert-base-uncased/` and `./bert-base-chinese/`, or update the corresponding YAML path.

## Training entry point

```bash
python train.py --config_file configs/train_mosi.yaml
python train.py --config_file configs/train_mosei.yaml
python train.py --config_file configs/train_sims.yaml
```

The manuscript's independent training runs use seeds `2024`, `2025`, and `2026`. The YAML default is `2024`; pass `--seed 2025` or `--seed 2026` for the other runs. These training seeds are distinct from validation/test mask seeds.

For every sample and epoch, corruption is regenerated reproducibly from the training seed, epoch, and sample index. Each epoch is followed by validation on:

- missing rates: `0.0`, `0.1`, `0.5`, `0.9`, `1.0`;
- mask seeds: `1111`, `2222`, `3333`.

The checkpoint with the lowest mean MAE across all 15 validation conditions is saved as:

```text
ckpt/<dataset>/best_validation_MAE_<training-seed>.pth
```

The test split is never used for checkpoint selection.

## Robustness evaluation entry point

Evaluation requires a checkpoint trained with this revised architecture.

```bash
# Continuous simultaneous corruption at eta = 0.5
python robust_evaluation.py \
  --config_file configs/eval_mosei.yaml \
  --ckpt_path ckpt/mosei/best_validation_MAE_2024.pth \
  --pattern continuous --missing_rate 0.5 --mask_seed 1111

# Complete text missingness
python robust_evaluation.py \
  --config_file configs/eval_mosei.yaml \
  --ckpt_path ckpt/mosei/best_validation_MAE_2024.pth \
  --pattern text_missing

# Complete audio + vision missingness
python robust_evaluation.py \
  --config_file configs/eval_mosei.yaml \
  --ckpt_path ckpt/mosei/best_validation_MAE_2024.pth \
  --pattern av_missing

# Contiguous 50% block missingness
python robust_evaluation.py \
  --config_file configs/eval_mosei.yaml \
  --ckpt_path ckpt/mosei/best_validation_MAE_2024.pth \
  --pattern block --missing_rate 0.5

# Independent corruption plus a contiguous burst
python robust_evaluation.py \
  --config_file configs/eval_mosei.yaml \
  --ckpt_path ckpt/mosei/best_validation_MAE_2024.pth \
  --pattern mixed_burst --missing_rate 0.2 --block_rate 0.3
```

Named asymmetric conditions are `text_heavy` (`0.7/0.1/0.1`) and `av_heavy` (`0.1/0.7/0.7`). Use `--text_rate`, `--audio_rate`, and `--vision_rate` to override any condition explicitly. The report prints both requested and realized text/audio/vision missing rates.

Evaluation defaults to the manuscript's server-side FP32 protocol. Use `--precision fp16` only for a compatible CUDA deployment such as the dedicated Jetson protocol.

## Lightweight verification

These checks do not train or evaluate the research model:

```bash
python -m unittest discover -s tests -v
```

Full end-to-end execution additionally requires the processed datasets, pretrained BERT weights, a compatible CUDA environment, `causal-conv1d`, and `mamba-ssm`.

## Reproducibility notes

- Keep dataset preprocessing, training seeds, mask seeds, and missingness patterns fixed when comparing models.
- Do not select checkpoints or decision thresholds using test results.
- Preserve the explicit distinction between padding, observed zero-valued features, and missing observations.
- At `eta = 1.0`, mask-eligible text becomes `[UNK]` and audio/vision are replaced by learned missing tokens after corruption.
- Hardware latency, memory, power, and temperature depend on the exact build and runtime environment.

## Citation

Until a public preprint or final bibliographic record is available:

```bibtex
@unpublished{hu2026cmsmamba,
  title  = {CMS-Mamba: Missing-Aware State-Space Stabilization for Robust Multimodal Sentiment Analysis under Incomplete Observations},
  author = {Hu, Jie and Dang, Qingxia and Li, Ming},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

Contact: Ming Li (`lettermail@wtu.edu.cn`).
