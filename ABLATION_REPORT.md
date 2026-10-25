# CipherBench — Controlled Ablation Study

Run `phase3-20261023T162551Z` completed 2026-10-23T16:28:41.714133+00:00.

## Protocol

- Implemented the same graph as Phase 1 (one 3×3 valid convolution, square activation, non-overlapping 2×2 average pool at stride 2, linear logistic-loss head). A0 is the Phase 1 baseline architecture/preprocessing control retrained under this common Phase 3 selection/refit protocol; Phase 1 saved weights and results remain unchanged. A1–A3 retrain under their stated configurations.
- A1 changes only the input representation to raw pixel /16 in [0,1]. A2 uses StandardScaler plus the selected squared-hinge range penalty. A3 combines pixel scaling and the selected range penalty. A4 is threshold tuning applied to A3 predictions only.
- Range term is `lambda * mean(max(abs(z)-B, 0)^2)` on convolution pre-activations, included in the training loss and differentiated analytically. No clipping is applied at inference.
- Candidate grid for StandardScaler: B ∈ {2,4,8}, λ ∈ {0.001,0.01,0.1}; pixel-scaled: B ∈ {0.25,0.5,1}, same λ values. Select using validation accuracy, positive F1, unpenalized validation log loss, then violation rate. Epoch selection uses validation only; final model refit uses the full outer training split for the selected epoch count.
- Primary benchmark uses the unchanged Phase 1 split (fit 214 / validation 38 / outer train 252 / test 108; seed 2026; split ID `digits01-seed2026-e1b8b781ea09ee9f`). It was previously seen in Phase 1 and 2 and is not a fresh confirmatory test. Digit 1 is positive.
- Secondary benchmark filters digits 3 and 8; mapping 3→0, 8→1; seed 3038; split ID `digits01-seed3038-b158100423688443`; fit 211 / validation 38 / outer train 249 / test 108. It is a new task, not a reproduction of the paper.
- Preprocessing for the StandardScaler path is fit on the fit partition during selection and refit on the complete outer training data for final evaluation. Pixel /16 is fixed by the source pixel intensity domain (0–16), not estimated from test data.
- Each test split was evaluated once after configuration/threshold freeze. Metrics use scikit-learn; confusion matrices are in class order [0,1]. Wilson intervals and all full metrics are in JSON.
- Training budgets: max 1000 epochs, patience 30, batch size 32, learning rate 0.001, L2 head coefficient 1e-4; all model seeds and actual selected epochs/configurations are stored in JSON. Training time measures selection-search plus final refit. Plaintext inference time was not separately benchmarked because the 64-feature graph is small and calls lack a stable timing harness.

## Primary benchmark: digits 0 versus 1

The test accuracy is saturated at 107/108 (99.07%) for all ablations. The single error moves between classes; no accuracy gain is observed. A3 and A1 produce the same test predictions, so this run does not isolate a benefit from range regularization.

| Model | Train acc | Val acc | Test acc | Positive P/R/F1 | Macro P/R/F1 | Confusion [0,1] | Test max |z| / violations |
|---|---:|---:|---:|---|---|---|---|
| A0 StandardScaler | 100.00% | 100.00% | 99.07% | 1.000 / 0.982 / 0.991 | 0.991 / 0.991 / 0.991 | `[[53, 0], [1, 54]]` | 14.995 / — |
| A1 pixel /16 | 100.00% | 100.00% | 99.07% | 0.982 / 1.000 / 0.991 | 0.991 / 0.991 / 0.991 | `[[52, 1], [0, 55]]` | 7.968 / — |
| A2 StandardScaler + range | 100.00% | 100.00% | 99.07% | 1.000 / 0.982 / 0.991 | 0.991 / 0.991 / 0.991 | `[[53, 0], [1, 54]]` | 14.403 / 1.11% |
| A3 RC-TinyCNN | 100.00% | 100.00% | 99.07% | 0.982 / 1.000 / 0.991 | 0.991 / 0.991 / 0.991 | `[[52, 1], [0, 55]]` | 5.325 / 76.26% |

Per-class test precision/recall:

| Model | Class 0 P/R | Class 1 P/R |
|---|---|---|
| A0 StandardScaler | 0.981 / 1.000 | 1.000 / 0.982 |
| A1 pixel /16 | 1.000 / 0.981 | 0.982 / 1.000 |
| A2 StandardScaler + range | 0.981 / 1.000 | 1.000 / 0.982 |
| A3 RC-TinyCNN | 1.000 / 0.981 | 0.982 / 1.000 |

The validation-selected A4 threshold was 0.0 (same as default), giving test accuracy 99.07% and the same predictions as A3. A0 versus A3 McNemar exact two-sided p=1.0 (two discordant pairs). A0 and A3 each scored 100% validation accuracy across three initialization/shuffle seeds on the same fixed partition (population SD 0); these are not independent split repetitions.

## Secondary benchmark: digits 3 versus 8

On this harder task, A1 and A3 reached 107/108 (99.07%); A0 and A2 reached 103/108 (95.37%). A1 equals A3 on this test set, so the observed difference is associated with pixel scaling in this experiment, not demonstrated range-penalty benefit. A2 does not improve the StandardScaler control.

| Model | Train acc | Val acc | Test acc | Positive P/R/F1 | Macro P/R/F1 | Confusion [0,1] | Test max |z| / violations |
|---|---:|---:|---:|---|---|---|---|
| A0 StandardScaler | 100.00% | 94.74% | 95.37% | 0.962 / 0.943 / 0.952 | 0.954 / 0.954 / 0.954 | `[[53, 2], [3, 50]]` | 15.145 / — |
| A1 pixel /16 | 100.00% | 100.00% | 99.07% | 0.981 / 1.000 / 0.991 | 0.991 / 0.991 / 0.991 | `[[54, 1], [0, 53]]` | 9.729 / — |
| A2 StandardScaler + range | 100.00% | 100.00% | 95.37% | 0.944 / 0.962 / 0.953 | 0.954 / 0.954 / 0.954 | `[[52, 3], [2, 51]]` | 9.019 / 4.06% |
| A3 RC-TinyCNN | 100.00% | 100.00% | 99.07% | 0.981 / 1.000 / 0.991 | 0.991 / 0.991 / 0.991 | `[[54, 1], [0, 53]]` | 7.817 / 85.03% |

Per-class test precision/recall:

| Model | Class 0 P/R | Class 1 P/R |
|---|---|---|
| A0 StandardScaler | 0.946 / 0.964 | 0.962 / 0.943 |
| A1 pixel /16 | 1.000 / 0.982 | 0.981 / 1.000 |
| A2 StandardScaler + range | 0.963 / 0.945 | 0.944 / 0.962 |
| A3 RC-TinyCNN | 1.000 / 0.982 | 0.981 / 1.000 |

A4 selected threshold 0.0. A0 versus A3 McNemar exact two-sided p=0.21875 (six discordant pairs), so this small sample does not establish a reliable performance difference. Validation accuracy over three initialization/shuffle seeds on the same split was A0 93.86% ± 1.24 percentage points and A3 99.12% ± 1.24 points; seeds vary initialization/shuffle only, not the split.

## Range and numerical diagnostics

No NaN/infinite values or magnitudes above 1e6 occurred in measured activation diagnostics. The test set diagnostics show that the selected penalty did not enforce its nominal bound: A3 bound B=1 had 76.26% of primary and 85.03% of secondary test pre-activations outside B. A2 violations were 1.11% (primary, B=8) and 4.06% (secondary, B=4). The pixel-only A1 max |z| was 7.968 / 9.729, versus A0 14.995 / 15.145. H1 has mixed, preliminary evidence: A3 reduced observed pre-square maxima relative to A0 and had lower validation CKKS-versus-float absolute errors at each measured stage. However, the selected B=1 was exceeded by most test pre-activations, so the penalty did not enforce the target range. This one validation run does not establish a general stability benefit or bound guarantee. Test range diagnostics are descriptive only and did not tune the models.

## CKKS validation and cost

CKKS was run on primary validation rows only (38), with Phase 2 exploratory parameters N=8192, coefficient modulus bits [49,40,40,40,49], global scale 2^40. One context/key generation was measured separately. One thread, batch size 1, no warm-up. Values include diagnostic intermediate decryptions and are not secure end-to-end serving measurements. The mask path decrypts pooled features before the plaintext head. The im2col path decrypts the square 6×6 map before plaintext pooling and head due to the installed TenSEAL vector API limitation; its privacy boundary differs and latency is not an apples-to-apples encrypted model comparison.

| Model/path | Agree with plaintext | Mean / worst max abs logit error | Mean seconds to plaintext prediction | Serialized bytes (first sample) |
|---|---:|---:|---:|---|
| A0_standard_scaler_control / mask_dot | 38/38 (0 flips) | 2.64e-05 / 7.44e-05 | 1.2847 | `{'input_ciphertext_bytes': 412023, 'convolution_ciphertexts_bytes_sum': 11275552, 'square_ciphertexts_bytes_sum': 7739364, 'pooled_ciphertexts_bytes_sum': 1039804, 'convolution_ciphertext_count': 36, 'square_ciphertext_count': 36, 'pooled_ciphertext_count': 9}` |
| A0_standard_scaler_control / im2col_partial | 38/38 (0 flips) | 2.14e-05 / 5.49e-05 | 0.0452 | `{'input': 411978, 'convolution': 313682, 'square': 214973}` |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / mask_dot | 38/38 (0 flips) | 1.72e-05 / 4.36e-05 | 1.3103 | `{'input_ciphertext_bytes': 411825, 'convolution_ciphertexts_bytes_sum': 11278933, 'square_ciphertexts_bytes_sum': 7739451, 'pooled_ciphertexts_bytes_sum': 1039537, 'convolution_ciphertext_count': 36, 'square_ciphertext_count': 36, 'pooled_ciphertext_count': 9}` |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / im2col_partial | 38/38 (0 flips) | 1.39e-05 / 3.28e-05 | 0.0467 | `{'input': 412760, 'convolution': 312891, 'square': 214934}` |

All four validation paths scored 38/38 and matched plaintext predictions 38/38. Mask-dot total includes diagnostic decryptions (mean ~1.285 s A0, ~1.310 s A3); im2col partial-path totals were ~0.0452 s and ~0.0467 s, respectively. The apparent latency difference is confounded by early decryption and different ciphertext representation. The mask path serializes many per-position ciphertexts (36 convolution and square values plus 9 pooled values), so listed byte sums are aggregate intermediate sizes, not wire bytes for a defined protocol. Full stage timings, means/medians/p95s, per-row errors, states, and sizes are in `artifacts/phase3_ckks_validation.json`.

### CKKS stage errors and timing (38 primary validation rows)

| Model/path | Stage | Mean per-sample MAE | Worst sample max absolute error | Mean stage seconds |
|---|---|---:|---:|---:|
| A0_standard_scaler_control / mask_dot | convolution | 1.25e-06 | 2.21e-06 | 0.958915 |
| A0_standard_scaler_control / mask_dot | square | 5.99e-06 | 1.22e-04 | 0.170105 |
| A0_standard_scaler_control / mask_dot | pooled | 1.25e-05 | 1.42e-04 | 0.010438 |
| A0_standard_scaler_control / mask_dot | logit | 2.64e-05 | 7.44e-05 | 0.000044 |
| A0_standard_scaler_control / im2col_partial | convolution | 3.28e-07 | 1.71e-06 | 0.030563 |
| A0_standard_scaler_control / im2col_partial | square | 9.16e-06 | 1.53e-04 | 0.004101 |
| A0_standard_scaler_control / im2col_partial | pooled | 9.09e-06 | 9.10e-05 | 0.000095 |
| A0_standard_scaler_control / im2col_partial | logit | 2.14e-05 | 5.49e-05 | 0.000095 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / mask_dot | convolution | 1.05e-06 | 1.48e-06 | 0.983472 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / mask_dot | square | 2.04e-06 | 1.25e-05 | 0.169865 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / mask_dot | pooled | 5.74e-06 | 2.35e-05 | 0.010512 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / mask_dot | logit | 1.72e-05 | 4.36e-05 | 0.000046 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / im2col_partial | convolution | 2.45e-07 | 1.34e-06 | 0.031744 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / im2col_partial | square | 5.99e-06 | 2.57e-05 | 0.004575 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / im2col_partial | pooled | 5.88e-06 | 1.95e-05 | 0.000088 |
| A3_pixel_scale_plus_range_penalty_RC_TinyCNN / im2col_partial | logit | 1.39e-05 | 3.28e-05 | 0.000088 |

The im2col stage-time grouping includes host-side layout in its encrypt/layout stage and plaintext pool/head in the last column; it is not a direct stage-by-stage equivalent to encrypted mask pooling. Decryption and one-time context/key setup remain separately recorded in JSON.
### A5 packing interpretation

TenSEAL im2col reduced the convolution-plus-square partial path time and preserved predictions on the validation rows, but could not retain encrypted pooling using the installed API route. It decrypts the square map early; hence it is a computational prototype only, not a privacy-equivalent optimization and not a complete encrypted inference result.

### Selected configurations and training time

| Task | Model | Selected epochs | Bound B | λ | Total training seconds | Max square output (test) |
|---|---|---:|---:|---:|---:|---:|
| digits_0_vs_1 | A0 StandardScaler | 1000 | — | 0 | 3.821 | 224.852 |
| digits_0_vs_1 | A1 pixel /16 | 993 | — | 0 | 4.502 | 63.494 |
| digits_0_vs_1 | A2 StandardScaler + range | 1000 | 8.0 | 0.001 | 7.717 | 207.442 |
| digits_0_vs_1 | A3 RC-TinyCNN | 999 | 1.0 | 0.001 | 4.615 | 28.351 |
| digits_3_vs_8 | A0 StandardScaler | 558 | — | 0 | 2.129 | 229.382 |
| digits_3_vs_8 | A1 pixel /16 | 1000 | — | 0 | 3.690 | 94.655 |
| digits_3_vs_8 | A2 StandardScaler + range | 1000 | 4.0 | 0.1 | 4.972 | 81.334 |
| digits_3_vs_8 | A3 RC-TinyCNN | 945 | 1.0 | 0.001 | 4.461 | 61.099 |

Full precision, recall, F1, macro scores, confusion matrices and class-wise scores for all fit/validation/test evaluations are stored in `phase3_ablation.json`; the tables above foreground test detail and train/validation accuracy. These numerical diagnostics use the final outer-train refit model and are descriptive.

## Interpretation and limitations

- H2 is descriptively supported on the secondary split: fixed pixel scaling improved 3-vs-8 test accuracy versus StandardScaler; on the primary task it made no accuracy difference. This is one small predefined split and does not establish generalization.
- H3 is not supported as a combined-method advantage: A3 ties A1 on both tests, and A3 ties A0 on primary accuracy. CKKS agreement is exact in class decisions for the validation samples, but both schemes use plaintext heads.
- Test sample counts are small (108). Primary has only one error; confidence intervals remain wide. The primary test split was repeatedly used in prior-phase diagnostics, so interpret it as a reconstruction check rather than an independent final claim.
- This reconstruction is not an exact reproduction of the paper: several training and model details were unspecified, and A1/A2/A3 are new controlled variants. The 49% reported CKKS result remains unresolved.
- The range bound is a penalty target, not a hard guarantee. A3’s large violation fraction means follow-up work would need stronger weights, different bound selection, or a constrained method chosen solely on development data. No such post-test tuning is performed here.

## Reproduction and artifacts

Activate `.venv`, then run `python scripts/run_phase3_ablation.py` to rerun all fit/validation selection, CKKS validation, and the one-time fixed-split test evaluation. To refresh CKKS measurements from already frozen weights without training or test access, run `python scripts/evaluate_phase3_ckks.py`.

- `artifacts/phase3_ablation.json`: configurations, train/validation/test metrics, timings, selected thresholds, range diagnostics, split references, and paired comparisons.
- `artifacts/phase3_models.json`: saved parameters and preprocessing metadata.
- `artifacts/phase3_splits.json`: indices, task mappings, seeds, and split IDs.
- `artifacts/phase3_predictions.csv`: test labels and predictions for all variants.
- `artifacts/phase3_ckks_validation.json`: CKKS per-sample errors, timings, ciphertext states/sizes, and prediction parity.
- `scripts/run_phase3_ablation.py`, `scripts/evaluate_phase3_ckks.py`; tests in `tests/test_phase3.py`.
