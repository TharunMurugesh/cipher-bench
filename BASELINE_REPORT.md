# CipherBench Baseline Reconstruction

**Run:** `baseline-20261023T152341Z-ea09ee9f`  
**Split:** `digits01-seed2026-e1b8b781ea09ee9f`  
**Scope:** Dataset pipeline, paper reference records, plaintext baseline controls, polynomial graph, and Phase 1 tests. No frontend, encrypted CNN, or proposed architecture was implemented.

## Method

The loader uses `sklearn.datasets.load_digits`, filters labels to 0 and 1, and retains the original 8×8 images, C-order row-major flattened 64-feature vectors, labels, and original dataset indices. There are 360 selected examples: 178 digit 0 and 182 digit 1. The label mapping is unchanged (`digit 0 → 0`, `digit 1 → 1`); positive class is 1 and metric/confusion-matrix class order is `[0, 1]`.

The frozen outer split is stratified 70/30 with seed 2026: 252 training examples (125 class 0, 127 class 1) and 108 held-out test examples (53 class 0, 55 class 1). The inner stratified validation partition contains 214 fit rows and 38 validation rows. The 108-example size is an outcome of this chosen reconstruction protocol, not evidence about the paper's unpublished split. Filtered positions and original `load_digits()` row indices are saved in [splits.json](artifacts/splits.json).

Per-feature `StandardScaler` is the paper-style preprocessing path. The Logistic Regression scaler and final TinyCNN scaler are fit on the outer training split only. The CNN's epoch selection uses a separate scaler fit on the inner fit split, then transforms validation data. Test data is only transformed with training-fitted values; test labels are used for the final held-out metrics and saved with predictions. The scaler uses population standard deviation (`ddof=0`); zero-variance feature scales remain 1. No alternative preprocessing was run and no clipping was added.

### Reconstruction choices and deviations

The primary paper describes scikit-learn digits, per-feature standardization, a 3×3 valid convolution, square activation, 2×2 average pooling, and a Logistic Regression head. It does not specify the TinyCNN kernel count, pooling stride, split procedure, or complete CNN training recipe. This implementation therefore uses one trainable convolution kernel, non-overlapping 2×2 average pooling with stride 2, and a binary logistic-loss head. The square activation is present during training. A NumPy/Adam implementation with batch size 32, learning rate `1e-3`, head L2 penalty `1e-4`, global gradient-norm clipping at 5, and validation-loss epoch selection is used; the chosen epoch count is then refit on the full outer training set. The validation loss continued improving through the configured 1,000-epoch cap, so the selected epoch hit that cap rather than an observed plateau. This is an explicit baseline reconstruction, not an exact reproduction.

The separate `PolynomialGraph` evaluates float64 valid convolution, square, stride-2 average pooling, and the fitted linear logit head using the same final weights as the plaintext TinyCNN. Its logits matched the TinyCNN's logits exactly on the held-out rows in this run (max/mean absolute difference 0; zero prediction flips). The graph is plaintext only; no encrypted inference is claimed here.

## Dataset and numerical diagnostics

| Diagnostic | Measured result |
|---|---:|
| Raw training pixel min / max | 0 / 16 |
| Raw training pixel mean / population standard deviation | 4.9170 / 6.0835 |
| Near-constant train features (`std ≤ 1e-8`) | 14 of 64 |
| Near-constant features on image border | 14 of 28 border features |
| Fitted per-feature scale min / max | 0.06287 / 7.36557 |
| Standardized training feature min / max | -3.9927 / 15.8430 |
| Convolution pre-square training activation min / max | -20.8557 / 10.0654 |
| Square activation training output min / max | 0.0000000053 / 434.9610 |
| Pooled feature training output min / max | 0.05927 / 158.6298 |
| NaN/infinite values across reported ranges | 0 |
| Intermediate values with absolute magnitude over `1e6` | 0 |

The exact counts, quantiles, per-feature indices, fitted scaler values, transformed test-input ranges, and activation range summaries are saved in [dataset_summary.json](artifacts/dataset_summary.json) and [baseline_results.json](artifacts/baseline_results.json).

## Held-out results

Metrics are calculated from the saved 108 true labels and predictions using scikit-learn 1.9.1. Positive class is digit 1. Confusion matrices use rows=true and columns=predicted in class order `[0, 1]`.

| Model / mode | Accuracy | Positive precision / recall / F1 | Macro precision / recall / F1 | Class 0 precision / recall | Class 1 precision / recall | Confusion matrix |
|---|---:|---|---|---|---|---|
| Majority baseline | 0.5093 | 0.5093 / 1.0000 / 0.6748 | 0.5093 / 0.5000 / 0.3374 | Not defined / 0.0000 | 0.5093 / 1.0000 | `[[0, 53], [0, 55]]` |
| Logistic Regression | 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | `[[53, 0], [0, 55]]` |
| Plaintext TinyCNN reconstruction | 0.9907 | 1.0000 / 0.9818 / 0.9908 | 0.9907 / 0.9909 / 0.9907 | 0.9815 / 1.0000 | 1.0000 / 0.9818 | `[[53, 0], [1, 54]]` |
| Plaintext-equivalent polynomial graph | 0.9907 | 1.0000 / 0.9818 / 0.9908 | 0.9907 / 0.9909 / 0.9907 | 0.9815 / 1.0000 | 1.0000 / 0.9818 | `[[53, 0], [1, 54]]` |

The majority model predicts digit 1 for every row because it is the majority class in the training split. Its class 0 precision is undefined and is stored as JSON `null`, not replaced by zero. Scikit-learn's `zero_division=np.nan` behavior is used; macro averages exclude undefined class metrics. Supports and full per-class metrics are in the JSON results.

## Timing

One full 108-row batch call was timed once with `time.perf_counter`; reported inference ms/sample is batch wall time divided by 108. There was no warm-up or repetition. Inference timing excludes preprocessing. Fit timing is model fitting only; TinyCNN timing includes validation selection plus its full outer-train refit, and excludes scaling. These small single-run CPU timings are diagnostic and are not directly comparable to the paper's reported timing scope.

| Model | Fit time (s) | Test batch model inference (ms/sample) |
|---|---:|---:|
| Majority baseline | 0.000094 | 0.000038 |
| Logistic Regression | 0.002654 | 0.001272 |
| Plaintext TinyCNN | 3.642438 | 0.002502 |
| Polynomial graph | Not applicable (uses fitted TinyCNN weights) | 0.001494 |

Outer scaler fit took 0.000368 s; outer train transform took 0.000123 s; outer test transform took 0.000085 s. The CNN inner-fit/validation scaler and transforms took 0.000477 s.

## Published reference and provenance

The primary paper's published results are stored separately in [published_reference.json](artifacts/published_reference.json) with `provenance_label: published_reference`. It reports `n=108` for its Logistic Regression table only; TinyCNN support and the exact split are not reported. Its precision, recall, F1, and confusion matrices are not reported and remain marked `Not reported`. The supporting Mali paper is theoretical; no method from it is implemented in Phase 1.

This run's Logistic Regression accuracy (1.00) and TinyCNN accuracy (0.9907) are local results on the chosen split. They do not establish exact reproduction or explain the paper's reported CKKS TinyCNN result.

## Reproducibility artifacts and commands

- Dataset and range diagnostics: [dataset_summary.json](artifacts/dataset_summary.json)
- Fixed split indices and identity: [splits.json](artifacts/splits.json)
- Metrics, predictions, timing, validation history, dependencies and limitations: [baseline_results.json](artifacts/baseline_results.json)
- Test labels and per-model predictions: [baseline_predictions.csv](artifacts/baseline_predictions.csv)
- Logistic and TinyCNN weights plus scaler parameters: [model_parameters.json](artifacts/model_parameters.json)
- Pinned environment: [requirements-phase1.txt](requirements-phase1.txt)

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_baselines.py
.\.venv\Scripts\python.exe -m unittest discover -v
```

The workspace has no Git repository metadata. No frontend or Phase 2/3 code was started.
