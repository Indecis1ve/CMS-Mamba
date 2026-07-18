# Paper-Aligned CMS-Mamba Design

## Goal

Bring the public CMS-Mamba implementation into structural and protocol-level alignment with the method described in `main_simple (6).tex`. The implementation must expose the paper's missingness information explicitly from data corruption through state-space discretization, and it must remove objectives and checkpoint-selection behavior that contradict the manuscript.

## Scope and Non-Goals

- Implement the paper-aligned data, model, training, validation, evaluation, configuration, and documentation paths.
- Treat existing checkpoints as intentionally incompatible with the revised parameter structure.
- Preserve the existing CLI entry points (`train.py` and `robust_evaluation.py`) where practical, while making their behavior match the manuscript.
- Add deterministic, lightweight unit and static tests that do not require the research datasets, a GPU, model training, or performance simulation.
- Do not download datasets or pretrained weights.
- Do not train, fine-tune, benchmark, simulate reported results, or claim numerical reproduction of the manuscript.
- Do not add unrelated architectural refactors.

## Source of Truth

The uploaded manuscript is authoritative for the following mechanisms and protocols:

1. Learnable Missing Modality Tokens (LMMT) for acoustic and visual frames.
2. Text-Aware Modality Mixing (TMM) with aligned missingness propagation.
3. Dynamic Time-Freezing (DTF), including the feature branch, explicit mask branch, non-negative base step, and fixed inference threshold `tau = 0.1`.
4. Separate AT- and VT-Mamba text-context sequences, averaged only before text-guided cross-attention.
5. Feature-wise acoustic/visual concatenation, key-only rotary position embedding (RoPE), bidirectional TQ-Mamba, adaptive max pooling, and pre-head Representation Normalization Lock (RNL).
6. MSE-only sentiment regression.
7. Per-sample shared training missing rate `eta ~ Uniform(0, 1)` and missingness-perturbed validation-only checkpoint selection.

## Architecture

The end-to-end path is:

```text
explicit corruption masks
  -> LMMT substitution on valid missing A/V frames
  -> TMM feature alignment and mask alignment
  -> AT-/VT-TC-Mamba with branch-specific DTF
  -> sequence-level text-context average
  -> feature-wise A/V concatenation
  -> key-only RoPE cross-attention
  -> bidirectional TQ-Mamba
  -> adaptive max pooling
  -> RNL LayerNorm
  -> linear sentiment regressor
```

The old reconstruction head and reconstruction loss are removed. The model class is named for CMS-Mamba, while `build_model(args)` remains the stable construction entry point used by the scripts.

## Data and Missingness Contract

### Mask semantics

All public model-facing missingness masks use one convention:

- `1.0`: the valid token or frame is missing.
- `0.0`: the position is observed or is padding.

Validity masks remain separate. A missingness mask must be a subset of its validity mask. This prevents genuine zero-valued frames and padding from being mistaken for missing observations.

Each sample provides:

- complete and corrupted text token tensors;
- complete and corrupted acoustic/visual feature tensors;
- text, acoustic, and visual validity masks;
- text, acoustic, and visual missingness masks.

Text corruption replaces only mask-eligible non-special tokens with token ID `100` (`[UNK]`). `[CLS]`, `[SEP]`, and padding remain uncorrupted. Both complete and corrupted text are encoded by the same configured BERT encoder.

### Training corruption

For each training sample and epoch, a deterministic random stream keyed by training seed, epoch, and sample index draws one shared `eta ~ Uniform(0, 1)`. Independent Bernoulli draws then corrupt eligible positions in all three modalities using that shared rate. No endpoint and no extra clean-sample fraction is forced.

The training dataset exposes `set_epoch(epoch)`, and `train.py` calls it before each epoch so corruption changes reproducibly between epochs without mutating the entire dataset from `__getitem__`.

### Validation and test corruption

Validation and test corruption are deterministic for a requested missing rate, mask seed, modality selection, and pattern. Validation checkpoint selection uses rates `0.0, 0.1, 0.5, 0.9, 1.0` and seeds `1111, 2222, 3333`, averages MAE across the resulting 15 settings, and never uses test performance.

Evaluation supports the manuscript's named families through explicit parameters:

- independent random missingness;
- contiguous block missingness;
- complete text missingness;
- complete acoustic and visual missingness;
- text-heavy asymmetric missingness;
- acoustic/visual-heavy asymmetric missingness;
- mixed-burst missingness;
- continuous simultaneous corruption for a caller-supplied rate.

Pattern parameters and realized rates are reported rather than hidden in code.

## TMM and Mask Alignment

Each CTC-inspired alignment module returns both:

- the aligned feature sequence; and
- a row-normalized alignment matrix `P` with shape `[batch, target_text_length, source_length]`.

After blank removal, probabilities are normalized across the source-time dimension so every non-empty row sums to one. Padding positions are excluded before normalization. Acoustic and visual masks are aligned using the same matrices:

```text
m_v_align = P_v @ m_v
m_a_align = P_a @ m_a
```

The aligned values remain continuous in `[0, 1]` and are not thresholded. TMM returns text, acoustic, and visual aligned features together with the aligned masks.

## LMMT

The model owns independent learnable acoustic and visual token vectors initialized from a small normal distribution. Substitution uses the explicit masks:

```text
x_stable = (1 - missing_mask) * x_corrupted + missing_mask * token
```

The validity mask is enforced so padding is never replaced by a learned token. No all-zero heuristic remains in the model.

## TC-Mamba and DTF

AT- and VT-Mamba remain separate TC branches. Their aligned missingness indicators are:

```text
m_AT[t] = concat(m_text[t], m_audio_align[t])
m_VT[t] = concat(m_text[t], m_vision_align[t])
```

For each stream in a TC branch, DTF computes:

```text
alpha = sigmoid(W_g x + W_m m + b_g)
delta_base = softplus(W_delta x + b_delta)
alpha_effective = alpha                         # training
alpha_effective = 1(alpha > 0.1) * alpha       # evaluation/inference
delta_effective = alpha_effective * delta_base
```

The same rule is applied to forward and backward selective scans. Backward inputs and masks are reversed together, and backward outputs are flipped back before fusion. Each stream has its own learned feature and mask projections. The implementation retains detached last-pass DTF statistics only for diagnostics; diagnostics never change the forward result.

TC-Mamba returns four sequences: acoustic output, visual output, AT text context, and VT text context. The two text contexts are averaged position-wise only after the TC stack.

## Cross-Attention, RoPE, TQ-Mamba, and RNL

Acoustic and visual sequences are concatenated along the feature dimension to produce `[B, L, 2d]`. The cross-attention module accepts text queries `[B, L, d]` and modal key/value inputs `[B, L, 2d]`.

- Queries use a learned projection from `d`.
- Keys and values use independent learned projections from `2d`.
- RoPE is applied to projected keys only.
- Values are not rotated.
- The attention result is projected back to `d` and combined with the projected/text residual at sequence level.

The per-head dimension must be even for RoPE; invalid configurations fail during model construction with a clear error.

The fused sequence passes through the configured bidirectional TQ-Mamba stack. Adaptive max pooling is applied only after TQ-Mamba. RNL is a standard affine `LayerNorm(d)` placed immediately before the linear regression head.

## Objective, Configuration, and Checkpoint Selection

Training uses only `MSELoss(sentiment_prediction, sentiment_target)`. Reconstruction modules, reconstruction configuration, and `alpha` loss weights are removed.

Dataset-specific model configurations are reconciled with the manuscript, including TC/TQ layer counts, state dimension, expansion, convolution size, attention heads, dropout, optimizer, learning rate, weight decay, batch size, epoch count, and DTF threshold. Configured feature dimensions are validated against loaded arrays with actionable errors instead of failing later in a matrix multiplication.

Each epoch performs training followed by the 15-condition validation grid. The model with the lowest mean validation MAE is saved. Test metrics may be computed only after training as reporting output and never influence checkpoint selection or filenames.

Checkpoint loading is strict by default. Because the user selected paper accuracy over backward compatibility, old checkpoints fail with an explicit incompatibility message rather than silently ignoring missing parameters.

## Error Handling and Invariants

The implementation validates:

- missing rates are in `[0, 1]`;
- masks have the expected batch/sequence shape and floating or boolean dtype;
- missing masks do not include padding;
- `[CLS]` and `[SEP]` remain observed during text corruption;
- alignment matrices and aligned masks contain finite values;
- non-empty alignment rows sum to one within tolerance;
- aligned missingness values remain in `[0, 1]` within numerical tolerance;
- acoustic and visual feature dimensions match configuration;
- DTF threshold is in `[0, 1)`;
- RoPE head dimensions are even;
- checkpoint state dictionaries match the revised model exactly.

Errors include the offending modality, expected shape/range, and actual value. Evaluation reports requested and realized missing rates for each modality.

## Verification Strategy

No model training, dataset download, research-data evaluation, throughput benchmark, or performance simulation is performed as part of this update.

Tests use deterministic small tensors and synthetic token/feature arrays only to verify code contracts:

1. Shared-`eta` corruption, boundary-token preservation, valid-position masking, deterministic seed behavior, and epoch variation.
2. LMMT replacement at missing valid positions only.
3. TMM alignment shape, row normalization, padding exclusion, and mask propagation.
4. DTF formula, non-negativity, mask-branch influence, training/evaluation threshold behavior, and zero-step behavior.
5. Forward/backward mask reversal consistency around the bidirectional scan adapter without claiming model-quality behavior.
6. Key-only RoPE behavior, feature-wise modal concatenation, output shape, and configuration validation.
7. MSE-only objective and absence of reconstruction outputs.
8. Validation-grid enumeration, mean-MAE checkpoint selection, and proof that test metrics cannot select a checkpoint.
9. Evaluation-pattern mask invariants and realized-rate reporting.
10. YAML schema and Python syntax/static import checks that are possible without CUDA-only Mamba kernels.

CUDA/Mamba integration tests are marked separately and skipped with an explicit reason when the required compiled dependencies are unavailable. A skipped integration test is not reported as evidence of end-to-end runtime success.

## Documentation and Delivery

The README is updated to describe the revised data contract, architecture, checkpoint incompatibility, exact CLI behavior, validation protocol, and verification limitations. Mojibake in user-facing source comments or output touched by this change is replaced with clear UTF-8 English text.

Changes are committed on `agent/paper-aligned-cms-mamba`, pushed to GitHub, and opened as a draft pull request targeting `main`. The pull request explicitly states that no training or paper-result reproduction was performed.

## Acceptance Criteria

- Every paper mechanism listed in the source-of-truth section has a direct, named implementation path.
- Explicit missingness masks flow from corruption through TMM into DTF.
- No zero-value missingness inference remains.
- DTF contains both feature and mask branches and uses `tau = 0.1` only in evaluation/inference.
- RoPE is applied only to modal keys.
- The training objective is MSE only.
- Checkpoint selection uses the 15-condition validation grid and never test metrics.
- Existing checkpoints are rejected clearly as incompatible.
- Lightweight tests and static checks pass in the available environment, with CUDA-only checks explicitly separated.
- No training, dataset evaluation, benchmark, or performance simulation is run.
- A draft GitHub pull request contains the complete code, configuration, tests, and documentation changes.
