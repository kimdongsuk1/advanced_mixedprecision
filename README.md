# advanced_mixedprecision

# tbd

1. mixed precision efficient usage of master weight: spliting last 8bit mantissa of master weight and copied for forward and bacward. merge before updating master weights. two scenarios are possible : bfloat training(bf16 & fp32) , e2m5(fp8 & fp16).

2. biased stochastic rounding : improved stochastic rounding. a master weight might not need.

The above two methods are implemented using cuda kernel.
