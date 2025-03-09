# advanced_mixedprecision

# tbd

# Efficient Mixed-Precision Training with Master Weights:

1. Optimized Master Weight Usage:

- The last 8-bit mantissa of the master weight is split and shared between the forward and backward passes.
- Before updating the master weights, the split parts are merged.

- This approach applies to two scenarios:
  * bfloat training: Uses BF16 for computation and FP32 for master weights.
  * e2m5 training: Uses FP8 (E2M5) for computation and FP16 for master weights.

2. Biased Stochastic Rounding:

- An improved stochastic rounding method.
- May eliminate the need for a master weight.

## Both methods are implemented using CUDA kernels.
