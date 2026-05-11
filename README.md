Simple GGML matrix bench. 

Main use to determinate supported instructions on NVIDIA CMP (Cryptocurrency Mining Processor) HX cards



### test hardware
|chip|name|short name| fp16, GFLOPS | fp32,GFLOPS 
|---|---|---|---|---|
|[Ampere](https://www.techpowerup.com/gpu-specs/?architecture=Ampere) [GA104](https://www.techpowerup.com/gpu-specs/nvidia-ga104.g964) | [NVIDIA CMP 70HX](https://www.techpowerup.com/gpu-specs/cmp-70hx.c3822)| HX | 10 710 |  10 710 | 
|[Ampere](https://www.techpowerup.com/gpu-specs/?architecture=Ampere) [GA104](https://www.techpowerup.com/gpu-specs/nvidia-ga104.g964) | [NVIDIA Quadro RTX A4000](https://www.techpowerup.com/gpu-specs/rtx-a4000.c3756)| A4 | 19 170 | 19 170 |

### ggml_add

matrix 512x512x256(512MB), 4096 iteration
mostly momory bound(A4)

| card | fp16, GFLOPS | fp32,GFLOPS | fp16/bandwidth | fp32/bandwidth | VRAM bandwidth |
|------|--------------|-------------|----------------|----------------|----------------|
| A4 | 63.80 | 31.95 | 382.8 | 373.52| 448(416 on my) |
| HX | 67.23 | 46.48 | 403.38 | 557.76 | 608.3 |
| A4,oc |77.23 | 39.48 | 463.38 | 473.76 | 512 |
| HX,oc |79.25 | 55.18 | 475.5| 662.16 | 726 |
