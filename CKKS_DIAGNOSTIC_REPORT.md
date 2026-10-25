# CipherBench CKKS Numerical Diagnostics

**Run date:** 2026-10-23 (Asia/Calcutta)  
**Scope:** CKKS feature extraction for the saved Phase 1 TinyCNN only. No retraining, proposed architecture, encrypted head, or frontend.  
**Environment:** Windows 11 x64, CPython 3.12.10, TenSEAL 0.3.18, NumPy 2.3.3, scikit-learn 1.9.1; CPU execution.

## Outcome

The mask-dot CKKS path was evaluated against the saved Phase 1 float64 polynomial graph. The paper's literal parameters create a context but fail during the square activation (`ValueError: scale out of bounds`); the available masked-convolution outputs are also numerically unusable. The requested `[60, 40, 40, 60]` diagnostic chain computes convolution and square accurately but has no remaining rescale level for encrypted multiplication by the average-pooling factor `0.25`.

A separately labeled three-rescale-level diagnostic chain, `[49, 40, 40, 40, 49]` with scale `2^40`, completes the same graph. It is accepted by TenSEAL at degree 8192 and its 218-bit total coefficient modulus. On the one final held-out evaluation, its 108 predictions matched the float64 graph exactly (zero flips) and achieved 107/108 accuracy (`0.9907`), the same as the local plaintext TinyCNN reconstruction. This demonstrates encrypted feature extraction with a **plaintext head**, not fully encrypted classification. It does not identify the cause of the paper's reported 0.49 accuracy.

## Audited plaintext computation

The implementation reads, without retraining or changing preprocessing, `tinycnn.parameters` and the saved TinyCNN scaler from `artifacts/model_parameters.json`. It checks the model and split IDs before running. The retained positive-class convention is digit `1` → label `1`; threshold is `logit >= 0`, equivalent to sigmoid probability `>= 0.5`.

1. Convert raw image pixels to the saved training-fitted per-feature standardized values: `(x - scaler_mean) / scaler_scale`. No clipping, refit, or alternative preprocessing is used.
2. Reshape the 64 features as an 8×8 image in C-order row-major layout.
3. Apply the single saved 3×3 kernel and bias at all 36 valid-convolution positions, in row-major output order. The convolution output shape is 1×6×6.
4. Square each convolution value. Shape remains 1×6×6.
5. Take non-overlapping 2×2 means with stride 2. The pooled shape is 1×3×3 and the head feature order is row-major (9 values).
6. Apply the saved 9-coefficient linear logit head and bias in plaintext after decrypting the pooled vector; predict digit 1 when the logit is nonnegative.

The model's square activation is part of its training graph and the saved exported parameters are used directly. This checks parity with the local reconstruction; it cannot establish how the original paper trained or exported its model.

## Operations and parameter behavior

TenSEAL's CKKS contexts report `auto_rescale`, `auto_mod_switch`, and `auto_relin` enabled. All three parameter chains created contexts successfully. The state and parameter IDs are saved for each operation in `artifacts/ckks_smoke.json`, `artifacts/ckks_validation.json`, and `artifacts/ckks_final_test.json`.

| Configuration | Context | Runtime graph result | Numerical / level result |
|---|---|---|---|
| Paper literal: degree 8192, `[40,21,21,40]`, scale `2^40` | Accepted | Stops at square activation with `ValueError: scale out of bounds` | Input ciphertext modulus count 3; mask-dot convolution count 2. On the development image, decrypted convolution mean absolute error was `1,049,997.22`, max `2,527,506.94`. No square, pool, or classification result exists. |
| Requested diagnostic: degree 8192, `[60,40,40,60]`, scale `2^40` | Accepted | Convolution and square complete; encrypted average pooling fails on `×0.25` with `ValueError: scale out of bounds` | Modulus counts: input 3 → convolution 2 → square 1. One-sample max errors: convolution `6.71e-7`, square `2.84e-5`. The failing pooling attempt and pre-scale ciphertext state are preserved. |
| Added-depth diagnostic: degree 8192, `[49,40,40,40,49]`, scale `2^40` | Accepted at total 218 bits | Convolution, square, and encrypted average pooling complete; pooled features are decrypted for the plaintext head | Modulus counts: input 4 → convolution 3 → square 2 → pooled output 1. Scale is reported as `2^40` at each successful stage. This adds one 40-bit rescale prime relative to the requested diagnostic chain while retaining its scale. |

For degree 8192, Microsoft SEAL's CKKS example documents a 218-bit maximum coefficient-modulus total and discusses scale stabilization with 40-bit intermediate primes. The added-depth chain reaches that documented bound and is accepted by the installed library. This is a parameter-limit check, **not an independent security audit or a formal security-level claim**. See the [SEAL CKKS example](https://github.com/microsoft/SEAL/blob/main/native/examples/5_ckks_basics.cpp). TenSEAL documents CKKS-vector dot products and encrypted/plain vector arithmetic; its convolution tutorial describes the distinct im2col approach, which was not substituted for this mask-dot implementation ([TenSEAL API and tutorial index](https://github.com/OpenMined/TenSEAL), [encrypted-convolution tutorial](https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb)).

The paper chain's context acceptance therefore does not imply that the requested neural-network graph is executable with the stated scale. TenSEAL does not throw during the paper chain's mask-dot step, but its decrypted convolution is already grossly wrong; the later square call throws. For the requested 40-bit diagnostic chain the mask-dot and square values are accurate to the recorded tolerances, while the exact encrypted pooling scale operation cannot be performed with the remaining chain.

## Numerical results

The third chain was first checked on one training-only fit sample. It produced max absolute errors of `1.75e-6` for convolution, `1.62e-5` for square, `1.71e-5` for pooled features, and `1.13e-5` for the plaintext-head logit. It preserved that sample's prediction.

All 38 saved validation rows were then evaluated with that unchanged configuration. There were no exceptions and no prediction flips. Accuracy and positive-class precision, recall, and F1 were each `1.0000` on this 38-row validation subset (19 examples in each class). Mean absolute errors averaged over each sample were `3.05e-7` convolution, `9.01e-6` square, `1.70e-5` pooled, and `4.01e-5` logit. The worst per-sample maximum errors were `1.41e-6`, `1.51e-4`, `1.65e-4`, and `1.01e-4`, respectively.

The configuration was frozen after validation. The saved 108-row test set was run once for Phase 2. Results from real saved labels and encrypted-feature predictions:

| Metric | Result |
|---|---:|
| Accuracy | 0.9907 (107/108) |
| Positive-class precision / recall / F1 (digit 1) | 1.0000 / 0.9818 / 0.9908 |
| Macro precision / recall / F1 | 0.9907 / 0.9909 / 0.9907 |
| Class 0 precision / recall | 0.9815 / 1.0000 |
| Class 1 precision / recall | 1.0000 / 0.9818 |
| Confusion matrix, rows true and columns predicted `[0,1]` | `[[53, 0], [1, 54]]` |
| Agreement with float64 polynomial graph | 108/108; 0 flips |

This matches the Phase 1 TinyCNN's saved confusion matrix and accuracy on the same split. It is local evidence for this reconstruction and parameter chain, not a reproduction of the paper's undocumented training and split details. The paper's 0.49 CKKS score remains unexplained.

### Input and activation range checks

The final test inputs use the saved training-fitted scaler. Standardized inputs ranged from `-3.3415` to `15.8430`, with mean `0.00449` and standard deviation `0.88791`. The raw pixel domain remains `[0,16]`; 14 of 64 features are near-constant training features, all at the border, as reported in Phase 1. On the final test rows, plaintext pre-square convolution ranged from `-14.9951` to `6.0432`; squared values ranged from approximately `1.07e-8` to `224.8518`; pooled values ranged from `0.07473` to `98.9545`; logits ranged from `-55.0324` to `67.2147`. All inspected values were finite; no intermediate exceeded absolute magnitude `1e6`. CKKS input encoding/decryption mean absolute error averaged `8.51e-10` per sample; the largest per-sample maximum was `8.66e-9`.

Final-test average per-sample stage errors were:

| Stage | Mean of sample MAE | Worst sample max absolute error |
|---|---:|---:|
| Convolution | `1.10e-6` | `2.87e-6` |
| Square | `1.46e-5` | `2.37e-4` |
| Average pool | `2.31e-5` | `1.81e-4` |
| Plaintext-head logit | `5.63e-5` | `1.27e-4` |

### Timing

The final run used one CPU process, one pass, no warm-up, and 108 separate samples. Context/key generation was performed once and is excluded. Stage timings include per-stage decryptions needed to record intermediate errors; the resulting timing is diagnostic and is not directly comparable to the paper's timing. Mean / median / p95 per sample were: encryption `6.47 / 6.00 / 9.23 ms`; encrypted convolution `738.07 / 741.25 / 805.15 ms`; encrypted square `152.62 / 149.60 / 187.72 ms`; encrypted pooling `10.38 / 9.73 / 14.67 ms`; intermediate decryptions `106.85 / 105.01 / 124.59 ms`; full measured feature extraction `1,017.11 / 1,018.99 / 1,116.74 ms`. The plaintext linear head, metrics, context creation, and key generation are outside that timer. These timings include 36 separate mask-dot outputs and their diagnostic decryptions; no im2col/packed optimization was run.

## 49% discrepancy hypotheses

The experiments changed only the CKKS parameter chain between the requested configurations; input preprocessing, saved weights, masks, operation order, pooling, class mapping, and threshold remained fixed.

1. **Scale, modulus chain, and rescaling:** Strongly implicated as a blocker in the literal setup. The paper chain produces grossly incorrect convolution values and throws at square. The requested 40-bit diagnostic chain has accurate square outputs but cannot rescale encrypted pooling. Adding one 40-bit middle prime at the same `2^40` scale completes the graph with small error. This does not prove that this was the cause of the paper's specific 0.49 score.
2. **Input and activation ranges:** Standardized inputs can reach `15.8430`; squared activations can reach `224.8518` on the held-out set. These ranges are finite and not unusually large by the Phase 1 diagnostic's `1e6` criterion. No clipping or alternative scaling experiment was performed, so range effects are not ruled out.
3. **Training/export of square activation:** The saved local TinyCNN is trained with square in its graph and exported weights match the plaintext graph. Whether the paper's original training/export did so cannot be verified from its reported details.
4. **Mask indexing and pooling alignment:** Row-major valid-convolution masks and 2×2 stride-2 pooling were compared position-by-position with the float64 reference; all tests pass with small CKKS error. The paper does not publish its exact indexing implementation.
5. **Labels and threshold:** Positive class is digit 1, class order is `[0,1]`, and the head thresholds its logit at zero. Encrypted predictions match the plaintext graph on all 108 final rows; no mapping or sign mismatch appears in this local path.
6. **Exceptions and invalid outputs:** Both requested parameter chains have preserved exceptions at their unsupported stages. The added-depth chain completes all 108 rows with no exception, NaN, or infinite value. A paper implementation's handling of an exception cannot be inferred from the article.

**Conclusion on 0.49:** The paper's listed parameters do not support the full mask-dot → square → encrypted-average-pool computation in this TenSEAL environment at scale `2^40`. Under a scale-preserving added-depth chain, the local model matches the plaintext graph and reaches 0.9907 accuracy. This makes an implementation/parameter mismatch a plausible research lead, but does not establish the cause of the published 0.49 result; the original weights, training details, exact split, and full executable are unavailable here.

## Reproduction, tests, and artifacts

Commands executed:

```powershell
.\.venv\Scripts\python.exe scripts\ckks_diagnostic.py --mode smoke
.\.venv\Scripts\python.exe scripts\ckks_diagnostic.py --mode validation
.\.venv\Scripts\python.exe scripts\ckks_diagnostic.py --mode final-test
.\.venv\Scripts\python.exe -m unittest discover -v
```

The final test run is recorded at `artifacts/ckks_final_test.json`; development, validation, and requested-parameter failures are in `artifacts/ckks_smoke.json` and `artifacts/ckks_validation.json`. Each result stores inputs, plaintext intermediate values, decrypted values, numerical errors, actual ciphertext scale/parameter IDs/modulus counts, exceptions with full tracebacks, predictions, metrics where applicable, and timings. The diagnostic script reuses Phase 1's split, scaler, weights, `PolynomialGraph`, and metrics library.

The full suite has **13 passing tests**: seven Phase 1 tests and six Phase 2 tests covering encrypted addition, ciphertext/plaintext weights, square, average pooling, complete feature extraction and numerical parity, plus the expected paper-chain square failure. No package was installed for Phase 2.

## Limitations

- The paper chain and 40-bit diagnostic chain do not complete the full graph at the specified scale; their exceptions are results, not suppressed errors.
- The added-depth configuration reaches the library's documented total-modulus limit. No independent security audit or formal security claim is made.
- The plaintext head requires decrypting pooled features, matching the paper's privacy boundary. The complete classifier is not end-to-end encrypted.
- Mask-dot execution is slow and diagnostic decryptions add latency. No packed/im2col optimization, encrypted linear head, or alternative input scaling was evaluated.
- The local model is a documented reconstruction, not the paper's known weights or exact training procedure. Therefore the origin of the reported 49% cannot be conclusively assigned.
