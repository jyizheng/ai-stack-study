nvcc -O3 \
  -gencode arch=compute_80,code=[compute_80,sm_80] \
  -gencode arch=compute_90,code=[compute_90,sm_90] \
  fused_lora.cu \
  -o fused_lora



