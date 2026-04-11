# Cachemir - FHE LLM Inference

CKKS-based fully homomorphic encrypted inference of generative LLMs with KV cache.

## Quick Start

```bash
# Quick verification
go run .

# Full version (main results)
go run . -logN=16 -hidDim=4096 -expDim=16384 -seqLen=512

# Bootstrapping placement algorithm
python3 bootstrap.py --prune=1

# Reproduce results (evaluates all modules, writes .csv, runs algorithm)
bash run_test.sh

# Homomorphic ops benchmark (level 1-16)
go run . -test=Ops -level=[level] -logN=16
```

## Test Commands

```bash
# Run all Stage1 tests
go test -v -run TestStage1

# Run specific tests
go test -v -run TestStage1_LinearPipelineMSE
go test -v -run TestStage1_NonlinearInPlaintext
go test -v -run TestStage1_DecoderPlaintext

# With custom config
CACHEMIR_CONFIG=myconfig.json go test -v -run TestStage1

# Relaxed tolerance (for debugging)
CACHEMIR_TOL=1e-3 go test -v -run TestStage1_LinearPipelineMSE

# Disable assertion mode (output only)
CACHEMIR_ASSERT=0 go test -v -run TestStage1_NonlinearInPlaintext
```

## Configuration System

Config file `config.json` controls model/crypto/runtime params. Flag params override config.

```bash
# Generate default config template
go run . -gen-config default_config.json

# Run with config
./cachemir_linear -config myconfig.json

# Flag overrides config
./cachemir_linear -config config.json -hidDim 64
```

Priority: CLI flags > config file > defaults

## Architecture

| File | Purpose |
|------|---------|
| `main.go` | CLI entry, test module dispatcher |
| `config.go` | Config loading/merging |
| `linear.go` | Linear layers (QKV, RoPE, AttnV, UpGate, Down, etc.) |
| `llama.go` | Decoder implementations (Moai/Thor/Plaintext) |
| `nonlinear_moai.go`, `nonlinear_thor.go` | Bootstrapping, Softmax, SiLU, Norm |
| `util.go` | Context prep, weight encoding, TestHelper, rotation utils |
| `linear_test.go` | Stage1 MSE-validated tests |

## Module Structure

- **Linear layers**: `Linear()` handles Q/K/V/Out (expand=0), Up/Gate (expand=1), Down (expand=-1)
- **Parallel mode**: `-parallel` flag enables multi-threaded evaluation via `runtime.GOMAXPROCS`
- **Key helper**: `rotateAnyStep()` for Galois rotations (binary decomposition)

## Known Issues (documented in problems.md)

1. `Down` projection has protocol bug (~0.09 RMSE) with `expand=-1` slot mapping
2. `NormThor` precision collapse in `nonlinear_thor.go:871` - InvSqrt pre-scaling fails at `hidDim=32`
3. `DecoderThor` unusable due to NormThor dependency
4. `Softmax/SiLU/Norm` panic if default `-level=6` is insufficient
5. `main.go` has incomplete test cases (Out/QK_T missing MSE checks)

## Crypto Parameters

- **Library**: Lattigo v6 (`github.com/tuneinsight/lattigo/v6`)
- **Scheme**: CKKS with bootstrapping
- **Key params**: `logN=8` (N=256), `logDefaultScale=41`, `logQ`/`logP` arrays
- Tests use level 6 by default; full runs use level 16

## Precision Expectations

- MSE threshold: `1e-5` default
- Precision formula: `-log2(sqrt(MSE))` bits
- General requirement: >15 bits for model correctness
- 20 bits ≈ 6 decimal places

## Documentation

All new feature documentation should be placed in the `docs/` folder.
