import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(X, Y, Z, N):
    pid = tl.program_id(axis=0)
    block_start = pid * 256
    offsets = block_start + tl.arange(0, 256)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    z = x + y
    tl.store(Z + offsets, z, mask=mask)

# Parameters
N = 1024
BLOCK_SIZE = 256
B = triton.language.constexpr(BLOCK_SIZE)

# Allocate input and output tensors on the GPU
X = torch.randn(N, device='cuda', dtype=torch.float32)
Y = torch.randn(N, device='cuda', dtype=torch.float32)
Z = torch.empty(N, device='cuda', dtype=torch.float32)

# Define grid size
grid = (triton.cdiv(N, BLOCK_SIZE),)

# Launch the kernel
vector_add_kernel[grid](X, Y, Z, N)

print("z:", Z)