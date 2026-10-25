# CipherBench Final Results

**Evidence freeze:** saved Phase 0–3 artifacts generated on 2026-10-23. Results below are descriptive measurements, not evidence of statistical significance.

## Research question and objectives

Can preprocessing and training changes make a shallow polynomial CNN more numerically suitable for CKKS while retaining classification quality? CipherBench audited the paper's reported CKKS result, reconstructed a plaintext model, tested CKKS feature extraction against its float64 graph, and compared standardization, fixed pixel scaling, and a training-only pre-activation range penalty on two binary digit tasks.

## Evidence and protocol

The primary task maps digit 0 to class 0 and digit 1 to positive class 1. The 360 filtered samples use a deterministic stratified 70/30 train/test split (252/108) and a validation subset taken only from the training partition. The Phase 1/2 primary split ID is `digits01-seed2026-e1b8b781ea09ee9f`; Phase 3 records the same primary test rows. The secondary task maps digit 3 to class 0 and digit 8 to positive class 1 and uses a separate seed-3038 split, ID `digits01-seed3038-b158100423688443` (249/108 train/test). Do not pool the tasks. Test predictions were produced after configurations were frozen. Phase 3 also reports three initialization/shuffle seeds on fixed development rows; those are repeated validation runs, not independent test splits.

All tables use the stored test predictions and labels in `artifacts/baseline_predictions.csv` and `artifacts/phase3_predictions.csv`, and metric records in `artifacts/baseline_results.json` and `artifacts/phase3_ablation.json`. Confusion matrices use row = actual, column = predicted, class order `[0, 1]`. Positive class is 1. Undefined precision is stored as JSON `null` and shown as `N/A`; macro scores use the project metric policy and are only reported when present.

## Published base-paper results (digits 0 vs 1)

The primary paper reports 1.00 accuracy for Logistic Regression in plaintext, CKKS and BFV (test support 108 for these rows). For its TinyCNN it prints 0.98 plaintext accuracy and 0.49 for “Encrypted Conv Features + LogReg”; the paper does not state TinyCNN test support, confusion matrix, precision, recall or F1. The TinyCNN figures are reproduced here as printed, not recomputed. Paper latency and ciphertext sizes are reported below with the paper's own workflow labels.

| Published method | Mode as reported | Accuracy | Positive/macro precision, recall, F1 | Latency (ms/sample) | Ciphertext bytes |
|---|---|---:|---|---:|---:|
| Logistic Regression | Plaintext | 1.00 | Not reported | 0.05 | 0 |
| Logistic Regression + polynomial sigmoid | CKKS | 1.00 | Not reported | 12.37 | 432,469 |
| Quantized Logistic Regression | BFV | 1.00 | Not reported | 2.55 | 432,468 |
| Conv + square + average pool + Logistic Regression | Plaintext | 0.98 | Not reported | 0.82 | 0 |
| Encrypted convolution features + Logistic Regression head | CKKS feature extraction; paper says pooled features are decrypted before head | 0.49 | Not reported | 2,055.09 | 432,526 |

Source: Deonise et al. (2025), Table 1 (journal p. 57) and Table 2 (journal p. 58). The narrative calls the CNN result 98.1%; the table prints 0.98, so this report uses the table value and notes the discrepancy. The paper's general abstract/introduction language describes end-to-end encryption, but its detailed CNN method states that pooled features are decrypted and sent to a plaintext Logistic Regression head. CipherBench follows that detailed boundary in its local CKKS path.

## Reproduced Phase 1 controls (digits 0 vs 1, held-out n=108)

| Model | Accuracy | Positive precision | Positive recall | Positive F1 | Macro P/R/F1 | Confusion `[0,1]` |
|---|---:|---:|---:|---:|---|---|
| Majority class (predict 1) | 0.5093 | 0.5093 | 1.0000 | 0.6748 | 0.5093 / 0.5000 / 0.3374 | `[[0,53],[0,55]]` |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 / 1.0000 | `[[53,0],[0,55]]` |
| Plaintext TinyCNN reconstruction | 0.9907 | 1.0000 | 0.9818 | 0.9908 | 0.9907 / 0.9909 / 0.9907 | `[[53,0],[1,54]]` |
| Float64 polynomial graph using same TinyCNN weights | 0.9907 | 1.0000 | 0.9818 | 0.9908 | 0.9907 / 0.9909 / 0.9907 | `[[53,0],[1,54]]` |

The polynomial graph is not encrypted; it evaluates the same conv → square → average-pool → linear-logit arithmetic in controlled float64 arithmetic. It provides the reference used for CKKS numerical comparisons.

Phase 1 local timing records are one fit or one full test-batch model call, without warmup/repetition. Model-only inference per sample was approximately 0.00127 ms for Logistic Regression, 0.00250 ms for TinyCNN and 0.00149 ms for the float64 graph; Logistic Regression's separately measured preprocessing was 0.00079 ms/sample. Training time was 0.00265 s for Logistic Regression and 3.642 s for TinyCNN (including validation selection and outer-train refit). These short single-call timings are diagnostic and not comparable to the paper's protocol.

## Phase 2 CKKS diagnostic and held-out inference

The literal paper chain `(40, 21, 21, 40)` with scale `2^40` created a TenSEAL context and passed addition, but multiplication produced unusable output (max absolute error 577,825.0092 in the Phase 0 smoke test). The local Phase 2 model instead used the explicitly recorded exploratory chain `[49, 40, 40, 40, 49]`, polynomial modulus degree 8192, and scale `2^40`; it was not represented as the paper's parameter set.

| Split/mode | CKKS→plaintext-head accuracy | Prediction agreement vs float64 graph | Mean absolute error: conv / square / pooled / logit |
|---|---:|---:|---|
| Validation, n=38 | 1.0000 | 38/38 | 2.23e-7 / 9.97e-6 / 1.90e-5 / 3.39e-5 |
| Held-out test, n=108 | 0.9907 | 108/108 | 1.10e-6 / 1.46e-5 / 2.31e-5 / 5.63e-5 |

The final-test CKKS predictions equal the plaintext TinyCNN predictions on 108/108 rows. Its held-out confusion matrix and classification metrics are `[[53,0],[1,54]]`, positive P/R/F1 = 1.0000 / 0.9818 / 0.9908, macro P/R/F1 = 0.9907 / 0.9909 / 0.9907.

The Phase 2 final-test artifact records mean per-sample stage MAE. Worst per-sample max absolute errors were 2.87e-6 (convolution), 2.37e-4 (square), 1.81e-4 (pooling), and 1.27e-4 (logit). Mean per-sample timings were 6.47 ms encrypt, 738.1 ms encrypted convolution, 152.6 ms encrypted square, 10.4 ms encrypted pooling, 106.8 ms diagnostic intermediate decrypt, and 1,017.1 ms through feature extraction. The complete result does not include serialized ciphertext-byte measurements. These measurements are one-thread, batch-size-one, no-warmup. The Phase 3 CKKS cost measurements below are validation-only and must not be presented as held-out test latency.

## Phase 3 ablations: digits 0 vs 1 (held-out n=108)

| Variant | Accuracy | Positive P/R/F1 | Macro P/R/F1 | Confusion `[0,1]` |
|---|---:|---|---|---|
| A0 standardization control | 0.9907 | 1.0000 / 0.9818 / 0.9908 | 0.9907 / 0.9909 / 0.9907 | `[[53,0],[1,54]]` |
| A1 pixel /16 only | 0.9907 | 0.9821 / 1.0000 / 0.9910 | 0.9911 / 0.9906 / 0.9907 | `[[52,1],[0,55]]` |
| A2 standardization + range penalty | 0.9907 | 1.0000 / 0.9818 / 0.9908 | 0.9907 / 0.9909 / 0.9907 | `[[53,0],[1,54]]` |
| A3 pixel /16 + range penalty (RC-TinyCNN candidate) | 0.9907 | 0.9821 / 1.0000 / 0.9910 | 0.9911 / 0.9906 / 0.9907 | `[[52,1],[0,55]]` |

A1 and A3 tie on test accuracy and all listed classification scores. Their predictions are identical; the records are identical on this primary split. A0/A2 also tie with one different error placement. The exact A0/A3 paired comparison is two discordant rows, one in each direction, exact two-sided p=1.0; this provides no evidence of a difference. All four variants were 100% accurate on the same 38-row validation partition; three fixed-row initialization/shuffle runs for A0 and A3 each retained 100% validation accuracy. These repeats do not provide independent test-split evidence.

## Phase 3 ablations: digits 3 vs 8 (held-out n=108)

| Variant | Accuracy | Positive P/R/F1 | Macro P/R/F1 | Confusion `[0,1]` |
|---|---:|---|---|---|
| A0 standardization control | 0.9537 | 0.9615 / 0.9434 / 0.9524 | 0.9540 / 0.9535 / 0.9537 | `[[53,2],[3,50]]` |
| A1 pixel /16 only | 0.9907 | 0.9815 / 1.0000 / 0.9907 | 0.9907 / 0.9909 / 0.9907 | `[[54,1],[0,53]]` |
| A2 standardization + range penalty | 0.9537 | 0.9444 / 0.9623 / 0.9533 | 0.9537 / 0.9539 / 0.9537 | `[[52,3],[2,51]]` |
| A3 pixel /16 + range penalty (RC-TinyCNN candidate) | 0.9907 | 0.9815 / 1.0000 / 0.9907 | 0.9907 / 0.9909 / 0.9907 | `[[54,1],[0,53]]` |

A1 and A3 again tie exactly, including predictions, on the saved test split. A0 and A2 do not improve over standardization without the range penalty. A0 vs A3 has six discordant paired predictions (one A0-correct/A3-wrong, five A0-wrong/A3-correct), exact two-sided p=0.21875; do not claim statistical significance. On fixed development rows, three initialization/shuffle runs yielded mean validation accuracy 0.9386 (population SD 0.0124) for A0 and 0.9912 (SD 0.0124) for A3. This is limited repeated-validation evidence, not repeated held-out testing.

## Numerical fidelity, latency and size

Phase 2 final CKKS test: 108/108 class decisions agree with the plaintext polynomial graph. Phase 3 validation (n=38, primary task only) also had 38/38 agreement for the four tested paths. The mean-of-sample logit MAE / worst-sample max logit error were 2.64e-5 / 7.44e-5 (A0 mask-dot), 1.72e-5 / 4.36e-5 (A3 mask-dot), 2.14e-5 / 5.49e-5 (A0 im2col), and 1.39e-5 / 3.28e-5 (A3 im2col). On that validation run, the `mask_dot` path averaged about 1.285 s (A0) / 1.310 s (A3) to plaintext prediction; `im2col_partial` averaged about 45.2 ms (A0) / 46.7 ms (A3). These timings are one-thread, batch-size-one, no-warmup validation measurements and include different operations at different decrypt boundaries.

Phase 3 serialized sizes record multiple ciphertext intermediates per sample: mask-dot stores 36 convolution, 36 square and 9 pooled ciphertext objects, totaling about 20.06 MB (A0) / 20.06 MB (A3) across those saved intermediates; im2col records input, convolution and square objects, totaling about 0.941 MB (A0) / 0.941 MB (A3) when those three stages are summed. These are artifact size sums, not network traffic measured in a full client/server protocol. The JSON artifacts retain per-sample bytes. The paper reports one ciphertext-size figure per its own workflow (432,526 bytes for CNN feature extraction); those numbers are not directly comparable to the local intermediate sums.

The Phase 3 `im2col_partial` workflow decrypts the squared 6×6 convolution map before pooling, so it exposes more intermediate information and does not provide privacy equivalence to mask-dot. The `mask_dot` workflow decrypts pooled features before the plaintext linear head. Neither executes the full classifier in ciphertext. The 38-row CKKS test results in Phase 3 are validation diagnostics only, not test results.

## Findings and limits

- On these saved splits, fixed pixel scaling improved the secondary task over the standardization control by 4.63 percentage points (99.07% vs 95.37%) and tied the range-regularized candidate. On the primary task, all A0–A3 achieved 99.07% accuracy.
- Range regularization has not demonstrated an additional accuracy benefit beyond pixel scaling: A1 and A3 tie on both held-out tasks. Moreover, the selected A3 bound was exceeded by 76.3% of primary and 85.0% of secondary held-out pre-square activations; the penalty did not enforce the chosen bound.
- CKKS prediction parity demonstrates numerical consistency for this frozen shallow model and observed inputs under the tested chain. It is not evidence that local CKKS accuracy is comparable to the paper's 49% under an equivalent configuration.
- The paper's split, learned weights, complete training recipe, exact software environment and detailed intermediate tensors are not available. Its approximately 49% CKKS TinyCNN result and our local measurements used configurations and workflows that are not fully equivalent.
- The local encrypted feature extractor uses a plaintext classification head. Its accuracy cannot establish that we reproduced or outperformed a fully encrypted end-to-end model.
- Digits 0/1 and 3/8 are small, single outer-split evaluations. Seed repetitions reuse fixed development rows, and there is no multi-split test analysis. Do not generalize these findings to broader image tasks.
- The supporting paper by Mali (2025) proposes generalized matrix-ring BGV/BFV/CKKS schemes; it is theoretical background for encrypted matrix operations, not an implementation used by CipherBench.

## Artifact map and references

- Baselines and primary split: `artifacts/baseline_results.json`, `artifacts/baseline_predictions.csv`, `artifacts/splits.json`, `artifacts/model_parameters.json`.
- CKKS: `artifacts/ckks_smoke.json`, `artifacts/ckks_validation.json`, `artifacts/ckks_final_test.json`, `artifacts/phase3_ckks_validation.json`.
- Phase 3: `artifacts/phase3_ablation.json`, `artifacts/phase3_predictions.csv`, `artifacts/phase3_splits.json`, `artifacts/phase3_models.json`.

**References**

1. C.-A. Deonise, A.-M. Roangheși, L.-I. Gheorghe, E. Simion, B.-C. Mocanu, D. Țurcanu, and F. Pop. “Homomorphic Encryption for Privacy-Preserving Machine Learning Inference.” *Annals of the Academy of Romanian Scientists, Series on Science and Technology of Information*, 18(2), 2025, pp. 50–61. DOI: [10.56082/annalsarsciinfo.2025.2.50](https://doi.org/10.56082/annalsarsciinfo.2025.2.50).
2. B. Mali. “Generalized BGV, BFV, and CKKS for Homomorphic Encryption over Matrix Rings.” IACR Cryptology ePrint Archive, 2025/972. [ePrint 2025/972](https://eprint.iacr.org/2025/972).
3. A. Benaissa, B. Retiat, B. Cebere, and A. Belfedhal. “TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption.” ICLR Workshop on Distributed and Private ML, 2021. [arXiv:2104.03152](https://arxiv.org/abs/2104.03152).
4. N. Jain, K. Nandakumar, N. Ratha, P. Pankanti, and U. Kumar. “Efficient CNN Building Blocks for Encrypted Data.” AAAI Workshop on Privacy-Preserving AI, 2021. [arXiv:2102.00319](https://arxiv.org/abs/2102.00319).
5. T. Ishiyama, T. Suzuki, and H. Yamana. “Highly Accurate CNN Inference Using Approximate Activation Functions over Homomorphic Encryption.” IEEE BigData Workshop, 2020. [arXiv:2009.03727](https://arxiv.org/abs/2009.03727).
6. M. Baruch et al. “Training Large Scale Polynomial CNNs for E2E Inference over Homomorphic Encryption.” 2023. [arXiv:2304.14836](https://arxiv.org/abs/2304.14836).
7. F. Al Hossain and T. Rahman. “A Training Framework for Optimal and Stable Training of Polynomial Neural Networks.” Preprint, 2025. [arXiv:2505.11589](https://arxiv.org/abs/2505.11589).
8. P. Kern et al. “Encrypted Neural Networks without Overflows.” Preprint, 2026. [arXiv:2605.23096](https://arxiv.org/abs/2605.23096).
