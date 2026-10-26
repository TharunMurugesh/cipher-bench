# CipherBench: Project Description

## Project at a glance

CipherBench is a research prototype and educational dashboard for studying a small convolutional neural network (CNN) under approximate homomorphic encryption. It asks whether practical choices in preprocessing and training can make a square-activation CNN easier to evaluate with CKKS arithmetic while preserving classification quality.

The project brings together three things:

1. A locally trainable comparison between a paper-style reconstructed CNN and a CipherBench variant.
2. Controlled experiments that separate the effect of input scaling from the effect of an activation-range training penalty.
3. Plaintext and partial CKKS inference measurements, with saved predictions, metrics, timing, numerical-error diagnostics, and dataset files for inspection.

CipherBench is an independent implementation and diagnostic study. It is not the paper authors' original code or weights, does not claim an exact reproduction, and does not currently provide end-to-end encrypted classification.

## Motivation and research question

In ordinary machine learning inference, a service often needs access to the input in plaintext. Homomorphic encryption offers a way to compute on encrypted values, but encrypted arithmetic is more constrained and expensive than ordinary arithmetic. CKKS supports approximate arithmetic on real-valued data; polynomial operations are a natural fit, while general nonlinear functions can be difficult or costly to evaluate directly.

This project uses a square activation, `z²`, because it is a polynomial operation. The question is whether scaling inputs into a known range and discouraging unusually large convolution activations during training can make the computation more suitable for CKKS, without making classification worse.

The work therefore studies two related but separate outcomes:

- **Classification quality:** do the model variants correctly distinguish the selected digit classes on held-out examples?
- **Encrypted-computation behavior:** how closely do selected CKKS feature-extraction paths match the corresponding plaintext computation, and what numerical errors and costs do they produce?

Prediction agreement is not the same as classification accuracy. A CKKS path can agree perfectly with a plaintext model and still reproduce that model's mistakes.

## Data and tasks

The experiments use the scikit-learn Digits dataset: 1,797 handwritten digit images represented as 8×8 grayscale arrays. Each pixel intensity is an integer from 0 through 16. The research pipeline uses two binary tasks:

| Task | Class mapping | Held-out test examples |
|---|---|---:|
| Digits 0 vs 1 | 0 → label 0; 1 → label 1 | 108 |
| Digits 3 vs 8 | 3 → label 0; 8 → label 1 | 108 |

The interactive Model Lab also supports digits 2 vs 7 and 4 vs 9. Those are additional class-pair tasks from the same Digits dataset, not independent data sources. The positive class is label 1 in the binary task. The dashboard dataset inspector shows class counts and sample rows and provides source and CSV links.

Source: [scikit-learn Digits dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

## Models and architecture

The primary CNN is intentionally small and follows a paper-style computation graph reconstructed from the published description:

```text
8×8 image
  → one valid 3×3 convolution
  → square activation (z²)
  → 2×2 average pooling, stride 2
  → linear logistic classification head
```

The convolution produces a 6×6 map, the square operation is applied elementwise, and pooling produces a 3×3 feature map. The nine pooled values feed the binary classification head. One convolution kernel is used in the controlled ablation configuration.

The project compares:

- **Paper Model / standardized control:** per-feature `StandardScaler` preprocessing, fitted on training data only.
- **CipherBench Model / pixel scaling:** fixed scaling by dividing raw pixel values by 16, mapping the known 0–16 range to 0–1, optionally combined with a range penalty selected on validation data.

The core convolution, square activation, pooling, and classification head remain the same. The project also evaluates a conventional Logistic Regression baseline and a majority-class baseline in its saved baseline experiment.

## Optimization and controlled ablations

The research experiment separates the changes into four variants so their contributions can be inspected independently:

| Variant | Preprocessing | Activation-range penalty |
|---|---|---|
| A0 | Per-feature standardization | Off |
| A1 | Pixel value divided by 16 | Off |
| A2 | Per-feature standardization | On; range and strength selected on validation data |
| A3 | Pixel value divided by 16 | On; range and strength selected on validation data |

For the range-aware variants, the training objective adds a soft penalty to the binary logistic loss and L2 head penalty:

```text
λ × mean(max(|z| − B, 0)²)
```

Here, `z` is the convolution output before the square activation, `B` is the target absolute range, and `λ` controls the penalty strength. Activations inside the selected range do not incur this penalty; larger values are penalized smoothly. This affects training only. It does not add a layer or operation to inference.

The research pipeline searches candidate bounds and penalty strengths using validation data. Standardized inputs use bounds 2, 4, and 8; pixel-scaled inputs use 0.25, 0.5, and 1.0; both use penalty strengths 0.001, 0.01, and 0.1. Candidates are ranked by validation accuracy, positive-class F1, unpenalized validation log loss, then fewer range violations. A separate validation-selected threshold variant is also recorded in the research artifacts.

The dashboard Model Lab uses a smaller task-specific run configuration: it tries the 9 pixel-scaled range/strength combinations, selects on validation accuracy, then positive-class F1, then validation loss, and reports held-out results for both comparison models.

## Training procedure

The research pipeline uses fixed, stratified partitions so both classes are represented in each subset. For each primary task, approximately 70% of examples form the outer training partition and 30% are held out for final testing. A validation partition is carved from the outer training data for epoch selection and configuration tuning. For the saved research ablation runs:

| Task | Fit | Validation | Outer train | Test |
|---|---:|---:|---:|---:|
| Digits 0 vs 1 | 214 | 38 | 252 | 108 |
| Digits 3 vs 8 | 211 | 38 | 249 | 108 |

The CNN is initialized and trained from scratch; it is not fine-tuned from pretrained weights. Core research training settings include mini-batches of 32, Adam with learning rate 0.001, L2 regularization of 0.0001 on the classification head, global gradient-norm clipping at 5, up to 1,000 epochs, and early stopping after 30 epochs without validation improvement. The best validation checkpoint is selected. After the model configuration and epoch count are selected, the research pipeline refits on the full outer training partition and evaluates on the frozen test partition.

Standardization statistics are fitted on the fit partition for model selection, then fitted on the outer training partition for the final refit. They are never fitted using test examples. Pixel scaling is fixed from the source data's known intensity range and has no learned parameters.

The main split uses seed 2026; the secondary split uses seed 3038. The primary-task split is also reused by earlier project phases, so the final result is a fixed-split evaluation rather than a fresh independent replication. The ablation pipeline varies initialization/shuffle seed across three runs for selected variants while keeping the data split and selected configuration fixed; this measures seed sensitivity, not split-to-split or dataset-to-dataset variability.

## Evaluation measures

Classification metrics are computed from held-out predictions and include:

- Accuracy
- Positive-class precision, recall, and F1
- Per-class and macro-averaged metrics
- Confusion matrix, with label order 0 then 1
- ROC-AUC where probability or decision scores are available in Model Lab
- Wilson 95% intervals for accuracy, positive precision, and positive recall in the research artifacts
- Exact paired McNemar comparison for A0 vs A3 on the shared test examples

The confusion matrix follows the conventional layout `[[true negatives, false positives], [false negatives, true positives]]`. Metrics are always tied to a dataset task and a split; a percentage from one task or split should not be generalized beyond it.

The CKKS evaluation also records prediction agreement with the corresponding plaintext graph, stage-level numerical error, timing information, and ciphertext states or serialized sizes where available. These encryption diagnostics are distinct from supervised classification metrics.

## Main saved classification results

The following values are from the saved research test splits, each containing 108 examples:

| Task | Variant | Accuracy | Correct | Confusion matrix |
|---|---|---:|---:|---|
| Digits 0 vs 1 | A0: standardized control | 99.07% | 107/108 | `[[53, 0], [1, 54]]` |
| Digits 0 vs 1 | A1: pixel scaling | 99.07% | 107/108 | `[[52, 1], [0, 55]]` |
| Digits 0 vs 1 | A2: standardized + range penalty | 99.07% | 107/108 | `[[53, 0], [1, 54]]` |
| Digits 0 vs 1 | A3: pixel scaling + range penalty | 99.07% | 107/108 | `[[52, 1], [0, 55]]` |
| Digits 3 vs 8 | A0: standardized control | 95.37% | 103/108 | `[[53, 2], [3, 50]]` |
| Digits 3 vs 8 | A1: pixel scaling | 99.07% | 107/108 | `[[54, 1], [0, 53]]` |
| Digits 3 vs 8 | A2: standardized + range penalty | 95.37% | 103/108 | `[[52, 3], [2, 51]]` |
| Digits 3 vs 8 | A3: pixel scaling + range penalty | 99.07% | 107/108 | `[[54, 1], [0, 53]]` |

On digits 0 vs 1, the four CNN variants tie on accuracy for this test split. On digits 3 vs 8, pixel scaling reaches 107/108 correct, whether or not the range penalty is added. The exact A0-vs-A3 paired test comparison is not statistically significant on either task (two-sided p=1.0 for digits 0 vs 1 and p=0.21875 for digits 3 vs 8). These small fixed test sets do not establish a general performance advantage. In particular, the digits 3 vs 8 result does not show an additional test-score benefit from the range penalty over pixel scaling alone.

On the primary digits 0 vs 1 task, the saved conventional Logistic Regression baseline scores 100% on the same 108-example test split; the majority baseline scores 50.93%. The CNN and its float64 polynomial graph score 99.07%. This comparison is useful context: CipherBench is investigating an encryption-compatible computation path, not claiming that its CNN outperforms every plaintext classifier.

## CKKS experiments and privacy boundary

The project uses TenSEAL to run selected CNN feature computations using CKKS approximate arithmetic. The exploratory CKKS parameters recorded in the artifacts are polynomial modulus degree 8192, coefficient-modulus bit sizes `[49, 40, 40, 40, 49]`, and global scale `2^40`.

Two partial paths are studied:

1. Encrypt the input and evaluate convolution, square activation, and average pooling on ciphertexts; decrypt the pooled features before applying the linear classification head in plaintext.
2. Evaluate convolution and square activation on ciphertexts, then decrypt the squared 6×6 feature map; perform pooling and the classification head in plaintext.

Thus, neither path encrypts the complete prediction process. The research validation diagnostics found that these paths matched the float64 model's predictions on all 38 primary-task validation examples for the tested A0/A3 configurations. The saved final-test CKKS run for the standardized primary-task model matched the plaintext graph on all 108 test predictions and had the same 107/108 classification accuracy. This is evidence of numerical/prediction parity for the tested setup, not evidence of end-to-end encrypted inference or a security certification. The chosen CKKS chain is explicitly exploratory and local timing and ciphertext size are parameter- and machine-dependent.

Reference implementation: [TenSEAL paper](https://arxiv.org/abs/2104.03152). The project compares its reconstruction with a published homomorphic-inference paper but does not claim a like-for-like reproduction of that paper's training process or reported scores.

## Dashboard and project experience

The React dashboard is intended for evaluators and learners who want to inspect the model design and evidence without reading every experiment script. Its main areas are:

- **Model Lab:** locally train the Paper Model and CipherBench Model on a selected digit-pair task, then inspect each model's metrics and confusion matrix in separate views.
- **Results:** explore saved plaintext, ablation, and CKKS comparisons, including task filters and detailed record information.
- **Optimization & Architecture:** compare the base and modified model stages and explain pixel scaling, range-aware training, and validation-based selection.

The Model Lab includes a dataset inspector with class counts and sample rows, a source link, and a CSV download. Results from a fresh local run are temporary and do not overwrite the checked-in research artifacts. GitHub Pages can display saved results and dataset files, but because it is static hosting, it cannot launch Python training jobs.

## Reproducing the project

### Dashboard

Requires Node.js 22 or newer:

```sh
npm ci
npm run dev
```

To build and preview the static dashboard:

```sh
npm run build
npm run preview
```

### Python experiments

The pinned research environment targets Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -v
```

Main experiment entry points:

```sh
python scripts/evaluate_baselines.py
python scripts/run_phase3_ablation.py
python scripts/evaluate_phase3_ckks.py
python scripts/run_dashboard_benchmark.py --task digits_0_vs_1 --output result.json
python scripts/inspect_dataset.py --task digits_0_vs_1
```

The research outputs are stored in `artifacts/` as JSON and CSV. They include configuration, split information, saved predictions and labels, metrics, selected-model records, CKKS numerical diagnostics, and timing/size measurements.

## Limitations and appropriate interpretation

- The core dataset is small and low-resolution. Results on four digit pairs still come from the same source dataset.
- The headline research results use one fixed split per task. The three-seed check keeps the split fixed and is not an independent data-split replication.
- The paper-style model is a reconstruction because some source training and implementation details are unspecified. Local results are not directly comparable to the paper's headline score.
- The project studies partial encrypted feature extraction; intermediate decryption and a plaintext head remain in the evaluated paths.
- CKKS timings, numerical errors, and serialized ciphertext sizes depend on parameters, software versions, and hardware. The recorded exploratory parameters are not a security audit.
- The observed accuracy differences do not support broad claims of superiority, statistical significance, or generalization to other datasets.

## Repository map

| Path | Purpose |
|---|---|
| `src/` | React dashboard, views, data transformation, and styles |
| `data/` | Digits data loading, task pairs, and reproducible split utilities |
| `models/` | TinyCNN, baseline classifiers, and plaintext polynomial graph |
| `metrics/` | Classification metrics and statistical comparison helpers |
| `he/` | CKKS context, encrypted feature operations, and numerical diagnostics |
| `scripts/` | Training, ablation, dataset inspection, and CKKS experiment entry points |
| `artifacts/` | Saved configurations, model results, predictions, split data, and CKKS outputs |
| `public/` | Static dashboard assets and downloadable data files |

## References

- C.-A. Deonise et al., “Homomorphic Encryption for Privacy-Preserving Machine Learning Inference,” *Annals of the Academy of Romanian Scientists*, 18(2), 50–61 (2025). [DOI: 10.56082/annalsarsciinfo.2025.2.50](https://doi.org/10.56082/annalsarsciinfo.2025.2.50).
- Scikit-learn, [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- A. Benaissa et al., “TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption.” [arXiv:2104.03152](https://arxiv.org/abs/2104.03152).
