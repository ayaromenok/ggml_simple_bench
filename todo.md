## ToDo

- create matrices which will fit to L1(128kb) and L2 (4MB)
- use one run for both fp16/fp32
- output only base statistic
- move command-line processing to separate file.
- check all fp16 ops:
    - https://github.com/ggml-org/ggml/pull/1121: sub, mul, div (how to check locked on HX MAD?)
    - https://github.com/ggml-org/ggml/pull/1125: abs, sgn, neg, step, tanh, gelu, silu, silu_back, gelu_quick, relu, sigmoid, hardsigmoid, exp, hardswish, leaky_relu, sqr, sqrt, sin, cos, log, clamp.
