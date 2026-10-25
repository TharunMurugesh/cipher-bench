# CipherBench Environment and Homomorphic-Encryption Feasibility

**Run date:** 2026-10-23 (Asia/Calcutta)  
**Scope:** Initial environment inspection and minimal CKKS arithmetic feasibility checks, performed before the model and dashboard experiments.

## Summary

- TenSEAL installed and imported successfully in an isolated, 64-bit Python 3.12.10 virtual environment on Windows 11.
- CKKS ciphertext encryption, addition, multiplication, and decryption worked with the documented Microsoft SEAL example parameter chain `[60, 40, 40, 60]`, degree 8192, and scale `2^40`. For the tested multiplication, max absolute error was `8.06e-7`.
- The paper's literal chain `[40, 21, 21, 40]` and scale `2^40` created a context and passed encrypted addition, but ciphertext multiplication returned grossly incorrect values (max absolute error `577825.01`). The test did not change or suppress this result.
- The paper's parameters are accepted by the installed library, so they are not syntactically invalid. Their total coefficient-modulus size is 122 bits, below Microsoft SEAL's documented 218-bit maximum at degree 8192. However, the tested 21-bit rescale prime(s) do not preserve useful multiplication accuracy with the paper's `2^40` global scale in this test.
- **Feasibility status:** basic real CKKS arithmetic is feasible on this CPU with a scale-compatible chain. The paper-parameter multiplication is not numerically usable as tested. This initial check did not test full TinyCNN inference; the later diagnostics in this repository document that separate implementation.

## Inputs reviewed

- Primary paper: Deonise et al., *Homomorphic Encryption for Privacy-Preserving Machine Learning Inference*, including its CKKS configuration and encrypted CNN feature-extraction sections. It states degree 8192, coefficient-modulus bit sizes `(40, 21, 21, 40)`, and global scale `2^40`.
- Supporting paper: Bence Mali, *Generalized BGV, BFV, and CKKS for Homomorphic Encryption over Matrix Rings*. It is theoretical and does not document a TenSEAL inference implementation.

The primary paper is available at [doi:10.56082/annalsarsciinfo.2025.2.50](https://doi.org/10.56082/annalsarsciinfo.2025.2.50). Supplied copies were used for source verification but are not redistributed with this repository.

## Environment inspection

Commands included:

```powershell
python --version
py -0p
Get-CimInstance Win32_OperatingSystem | Select Caption,Version,BuildNumber,OSArchitecture
Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select TotalPhysicalMemory
nvidia-smi
Get-ChildItem -Force
git status --short --branch
python -c "import importlib.util; print({x: bool(importlib.util.find_spec(x)) for x in ['tenseal','numpy','pandas','sklearn','torch']})"
# Optional source metadata checks; paper copies are not included in this repository.
pdfinfo '<downloaded-primary-paper>.pdf'
pdfinfo '<downloaded-supporting-paper>.pdf'
```

Observed:

| Item | Observation |
|---|---|
| OS | Microsoft Windows 11 Pro, build 26300, 64-bit |
| Python | CPython 3.12.10, 64-bit, Microsoft C runtime 1943 |
| CPU | Intel Core i7-13700HX, 16 cores / 24 logical processors |
| RAM | 16,886,128,640 bytes reported by WMI (about 15.7 GiB) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB reported; driver 616.92 / CUDA UMD 13.4 |
| Initial project state | Research briefs and source papers; no implementation source or package manifest was present |
| Existing global Python dependencies | TenSEAL, NumPy, pandas, scikit-learn, and PyTorch were absent from the selected system Python |
| Docker | Docker command exists; no container was needed because the native wheel installed and ran |

The GPU was inspected but not used. TenSEAL's CPU CKKS path was used for this smoke test.

## Isolated environment and pinned dependencies

Project environment: `.venv` using CPython 3.12.10. The dependency pins are in [requirements-phase0.txt](requirements-phase0.txt):

```text
tenseal==0.3.18
numpy==2.3.3
```

TenSEAL's [PyPI installation instructions](https://pypi.org/project/tenseal/) document `pip install tenseal` and publish a CPython 3.12 Windows x64 wheel. The equivalent pinned installation used here was:

```powershell
uv venv --python <Python-3.12-executable> .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-phase0.txt
```

`uv` created the virtual environment without `pip`; the first `python -m pip` attempt consequently returned `No module named pip`. This was resolved by bootstrapping pip inside that same environment and installing the pinned file with pip:

```powershell
uv pip install --python .venv\Scripts\python.exe pip==25.2
.\.venv\Scripts\python.exe -m pip install -r requirements-phase0.txt
.\.venv\Scripts\python.exe -m pip freeze
```

Installed runtime package versions: TenSEAL 0.3.18, NumPy 2.3.3. The environment's pip tool is 25.2. `pip freeze` listed the two runtime dependencies above. The venv is ignored in [.gitignore](.gitignore).

## CKKS smoke test

Reproducible script: [scripts/phase0_smoke.py](scripts/phase0_smoke.py)

Command:

```powershell
.\.venv\Scripts\python.exe scripts\phase0_smoke.py
```

For both parameter sets the script creates a context, enables/observes TenSEAL's default `auto_rescale`, `auto_mod_switch`, and `auto_relin` flags, encrypts `[1.25, -2.0]` and `[0.5, 3.0]`, adds and multiplies the ciphertext vectors, decrypts the results, and computes maximum and mean absolute errors. Expected addition is `[1.75, 1.0]`; expected product is `[0.625, -6.0]`.

### Paper parameters, run literally

| Parameter | Value |
|---|---:|
| Scheme | CKKS |
| Polynomial modulus degree | 8192 |
| Coefficient-modulus bit sizes | `[40, 21, 21, 40]` |
| Total coefficient-modulus bits | 122 |
| Global scale | `2^40` = 1,099,511,627,776 |
| Context creation | Succeeded; no parameter exception |
| Encrypted addition | `[1.7500000006, 1.0000000007]`; max abs error `7.13e-10` |
| Ciphertext multiply | `[-348140.9070, 577819.0092]` vs `[0.625, -6.0]`; max abs error `577825.0092`; mean abs error `462983.2706` |
| Ciphertext coefficient-modulus count | 3 before multiplication; 2 after multiplication (one level consumed by auto-rescale) |
| Reported ciphertext scale | `2^40` before and after the multiply |
| Exception | None during the completed paper-parameter add/multiply/decrypt test |

The context accepted the paper's literal parameters, and this chain has 122 total coefficient-modulus bits. Microsoft's [SEAL CKKS example](https://github.com/microsoft/SEAL/blob/main/native/examples/5_ckks_basics.cpp) states the degree-8192 maximum is 218 bits. Thus the tested multiplication failure is numerical, not a context-construction/security-bound rejection. In the observed run, the encoded scale is `2^40` while the chain's middle primes are only 21 bits; TenSEAL auto-rescaling consumes one modulus level, but the decrypted multiplication is unusable. No parameters were silently substituted.

### Separately tested scale-compatible reference configuration

This is an explicitly separate diagnostic, not a claim that the paper used these values. It follows the parameter example in the official [Microsoft SEAL CKKS basics example](https://github.com/microsoft/SEAL/blob/main/native/examples/5_ckks_basics.cpp):

| Parameter | Value |
|---|---:|
| Scheme | CKKS |
| Polynomial modulus degree | 8192 |
| Coefficient-modulus bit sizes | `[60, 40, 40, 60]` |
| Total coefficient-modulus bits | 200 |
| Global scale | `2^40` = 1,099,511,627,776 |
| Context creation | Succeeded |
| Encrypted addition | `[1.7499999994, 0.9999999994]`; max abs error `5.92e-10` |
| Ciphertext multiply | `[0.6250000832, -6.0000008068]`; max abs error `8.07e-7`; mean abs error `4.45e-7` |
| Ciphertext coefficient-modulus count | 3 before multiplication; 2 after multiplication |
| Reported ciphertext scale | `2^40` before and after the automatically rescaled multiply |
| Exception | None |

TenSEAL exposes the ciphertext's scale, modulus-count, and parameter ID through the ciphertext object. This report records the directly observed scale and level-count change. No CNN-level schedule or multi-layer depth was tested.

CKKS ciphertexts use randomized encryption, so the last decimal places and small addition errors vary across script runs. The tables record the final run above; the qualitative multiplication outcomes were unchanged on the repeated run.

## Outcome and blockers

- **TenSEAL install/import:** passed on native Windows x64 with CPython 3.12.10; no container or fallback was needed.
- **CKKS encryption/add/decrypt:** passed with both tested parameter chains.
- **CKKS ciphertext multiplication:** accurate with `[60, 40, 40, 60]`; numerically failed with the paper's `[40, 21, 21, 40]` at `2^40` global scale.
- **Security/parameter gate:** both contexts were accepted by the installed library; the paper chain totals 122 bits, and the alternative totals 200 bits, both below SEAL's cited 218-bit limit for degree 8192. This is a library limit check, not an independent cryptographic security audit.
- **Encrypted inference:** simple encrypted arithmetic is demonstrably feasible. Full encrypted CNN inference, runtime, and prediction fidelity remain untested and must not be claimed.
- **Other blocker:** `uv venv` omitted pip by default. Pip 25.2 was seeded in the venv and the pinned install was then confirmed by pip; no remaining setup blocker.
- **Scope note:** this report records an initial feasibility gate only. The dataset, models, CKKS diagnostics, and dashboard are documented in the repository's later research reports.

