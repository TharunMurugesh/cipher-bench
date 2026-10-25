# Research Methodology

## Questions and scope

CipherBench investigates whether the primary paper's low TinyCNN CKKS accuracy can be diagnosed from its disclosed details; whether a plaintext reconstruction gives a reproducible reference; whether input scaling and training-time activation-range regularization change held-out classification on two small digits tasks; and how CKKS feature extraction trades numerical fidelity, runtime and intermediate-data exposure. The work does not establish a fully encrypted end-to-end classifier.

## Dataset and mappings

Source: `sklearn.datasets.load_digits` (1,797 8×8 grayscale images, integer intensities 0–16). The primary binary task filters digits 0 and 1 (360 images) and maps 0→class 0, 1→class 1. The secondary task filters digits 3 and 8 (357 images) and maps 3→class 0, 8→class 1. In both tasks, class 1 is positive. The original 8×8 image is retained for convolution; flattened 64-element C-order row-major values are used by vector preprocessing and linear models.

## Splits and leakage controls

Each task uses a fixed stratified 70/30 outer train/test split and saves both filtered-dataset and original sklearn row indices. Primary seed 2026 and split ID `digits01-seed2026-e1b8b781ea09ee9f`; secondary seed 3038 and saved split ID `digits01-seed3038-b158100423688443`. Within the outer training partition, a stratified 15% validation split selects TinyCNN stopping epoch/configuration. Test labels are excluded from fitting, preprocessing estimation, model selection and threshold selection. Phase 3 uses a separate task-specific split and freezes configurations before one final evaluation on each test set. The primary test partition had already been used in Phase 1/2 diagnostics, so it is not a new independent replication of that earlier analysis.

## Preprocessing and models

The Phase 1 paper-style control is per-feature `StandardScaler`; every scaler is fitted only on the corresponding training rows, then applied to validation/test rows. Logistic Regression uses `C=1`, `lbfgs`, `max_iter=2000`. The TinyCNN reconstruction uses one 3×3 valid convolution, square activation, 2×2/stride-2 average pooling, flattening and a linear logit head with logistic loss. Model selection uses validation loss and early stopping; details absent from the paper (single kernel, optimizer, seed, batch size, regularization and selection protocol) are local reconstruction choices, not exact reproduction claims.

Phase 3 ablations use fixed pixel preprocessing `x/16` (intensities in [0,16]) versus train-fitted standardization. A2/A3 add the training-only squared-hinge penalty `λ · mean(max(|z|−B, 0)^2)` on convolution pre-activations `z`, with analytic gradient and hyperparameters selected on validation rows. It is not an inference-time layer or operation. A1 is pixel scaling without range loss; A3 adds this training objective. The inference graph remains 3×3 convolution → square → 2×2 average pooling → linear classification head.

The float64 polynomial graph runs the same fitted TinyCNN weights and arithmetic and is used as the CKKS numeric reference. It is plaintext-equivalent evaluation, not encrypted evaluation.

## CKKS implementation and boundary

TenSEAL 0.3.18 / Microsoft SEAL CKKS was evaluated on CPU. Phase 0 tested the paper's literal degree 8192 chain `[40,21,21,40]` at scale `2^40` without parameter substitution: context/addition succeeded, multiplication decrypted with maximum absolute error 577,825.0092. A separate scale-compatible smoke chain `[60,40,40,60]` gave max multiply error 8.07e-7. Phase 2/3 used the recorded exploratory chain `[49,40,40,40,49]`, degree 8192, scale `2^40`; it is distinct from the paper chain.

The Phase 2 `mask_dot` computation encrypts the 64 inputs, computes 36 convolution outputs for one kernel using mask-based dot products, squares them, and performs 2×2 average pooling in CKKS. It decrypts pooled features before the linear classification head, which runs in plaintext. In Phase 3, the partial `im2col` route encrypts an im2col layout, computes convolution and square in CKKS, then decrypts the squared 6×6 map before plaintext pooling/head. These are encrypted feature-extraction paths with plaintext boundaries, not full encrypted classification. Diagnostic intermediate decrypts are also included in the reported timings where specified.

## Metrics, timing and reproducibility

Held-out labels and actual predictions are in CSV; metric JSON includes accuracy, positive-class precision/recall/F1, macro metrics, per-class metrics and confusion matrices in class order `[0,1]`. Undefined class precision/recall are kept as null. CKKS numerical errors are compared stage-by-stage to the float64 graph. Test and validation are kept distinct: Phase 2 used saved validation/test rows; Phase 3 CKKS fidelity and timing were measured on primary-task validation only (38 rows). All timing/ciphertext results are hardware- and path-specific; Phase 3 uses one thread, batch size one, no warmup.

To reproduce the checks on this Windows project:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
npm ci
npm test
npm run build
npm run dev
```

The dashboard reads the frozen `artifacts/` JSON/CSV files and published-reference JSON; it does not retrain or write results. Python runtime pins are listed in `requirements-phase0.txt` and `requirements-phase1.txt`; package versions are recorded in the environment and result reports.

## References

1. Deonise et al. (2025), “Homomorphic Encryption for Privacy-Preserving Machine Learning Inference,” DOI [10.56082/annalsarsciinfo.2025.2.50](https://doi.org/10.56082/annalsarsciinfo.2025.2.50).
2. B. Mali (2025), “Generalized BGV, BFV, and CKKS for Homomorphic Encryption over Matrix Rings,” IACR Cryptology ePrint Archive, [2025/972](https://eprint.iacr.org/2025/972).
3. Benaissa et al. (2021), TenSEAL, [arXiv:2104.03152](https://arxiv.org/abs/2104.03152).
4. Jain et al. (2021), Efficient CNN Building Blocks for Encrypted Data, [arXiv:2102.00319](https://arxiv.org/abs/2102.00319).
5. Ishiyama et al. (2020), Approximate activation functions over HE, [arXiv:2009.03727](https://arxiv.org/abs/2009.03727).
6. Baruch et al. (2023), Polynomial CNNs for end-to-end HE inference, [arXiv:2304.14836](https://arxiv.org/abs/2304.14836).
7. Al Hossain & Rahman (2025), stable polynomial NN training, [arXiv:2505.11589](https://arxiv.org/abs/2505.11589).
8. Kern et al. (2026), encrypted networks without overflows, [arXiv:2605.23096](https://arxiv.org/abs/2605.23096).
