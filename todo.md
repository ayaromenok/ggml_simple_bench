## ToDo

- use one run for both fp16/fp32
- add possibility to use multiply operands separated by coma
- output only base statistic
- check all fp16 ops:
    - https://github.com/ggml-org/ggml/pull/1125: abs, sgn, neg, step, tanh, gelu, silu, silu_back, gelu_quick, relu, sigmoid, hardsigmoid, exp, hardswish, leaky_relu, sqr, sqrt, sin, cos, log, clamp.
- add CPU and Vulkan build
- ~~test big 1D/2D matrix~~
