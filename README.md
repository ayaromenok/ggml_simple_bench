Simple GGML matrix bench. 

Main use to determinate supported instructions on NVIDIA CMP (Cryptocurrency Mining Processor) HX cards



### test hardware
|chip|name|short name|
|---|---|---|
|(https://www.techpowerup.com/gpu-specs/?architecture=Ampere)[Ampere] (https://www.techpowerup.com/gpu-specs/nvidia-ga104.g964)[GA104]|(https://www.techpowerup.com/gpu-specs/cmp-70hx.c3822)[NVIDIA CMP 70HX]| HX|
|(https://www.techpowerup.com/gpu-specs/?architecture=Ampere)[Ampere] (https://www.techpowerup.com/gpu-specs/nvidia-ga104.g964)[GA104]|(https://www.techpowerup.com/gpu-specs/rtx-a4000.c3756)[NVIDIA (Quadro) RTX A4000]| A4|

### ggml_add

matrix 1024x1024x512, 1024 iteration
mostly momory bound(A4)

|| card|| fp16, GFLOPS||fp32,GFLOPS||
|---|----|---|
|A4| 63.80| 31.95|
|HX|67.23|46.48 |
|A4,oc|77.23|39.48|
|HX,oc|79.25|55.18|