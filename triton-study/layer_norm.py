import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    gamma_ptr,  # gamma 参数指针
    beta_ptr,  # beta 参数指针
    mean_ptr,  # 均值指针
    var_ptr,  # 方差指针
    n_elements,  # 归一化的维度大小
    eps,  # 防止除零的小常数
    BLOCK_SIZE: tl.constexpr,  # 每个线程块处理的元素数量
):
    # 获取当前线程的索引
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)

    # 计算局部和
    local_sum = tl.sum(x, axis=0)
    local_sum_sq = tl.sum(x * x, axis=0)

    # 使用原子操作累加全局和
    tl.atomic_add(mean_ptr, local_sum)
    tl.atomic_add(var_ptr, local_sum_sq)

    # 同步所有线程
    tl.debug_barrier()

    # 加载全局和
    global_sum = tl.load(mean_ptr)
    global_sum_sq = tl.load(var_ptr)

    # 计算全局均值和方差
    global_mean = global_sum / n_elements
    global_var = (global_sum_sq / n_elements) - (global_mean * global_mean)

    # 归一化
    y = (x - global_mean) / tl.sqrt(global_var + eps)
    y = y * gamma + beta

    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)

def layer_norm_triton(x, gamma, beta, eps=1e-5):
    # 输入形状
    n_elements = x.shape[-1]
    assert n_elements == gamma.shape[0] and n_elements == beta.shape[0], "gamma and beta must match the last dimension of x"

    # 分配输出和中间结果
    y = torch.empty_like(x)
    mean = torch.zeros(1, device=x.device)
    var = torch.zeros(1, device=x.device)

    # 定义线程块大小
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    # 启动 Triton 内核
    grid = (x.numel() // n_elements,)
    layer_norm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        gamma_ptr=gamma,
        beta_ptr=beta,
        mean_ptr=mean,
        var_ptr=var,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y

# 创建输入张量和参数
x = torch.randn(32, 197, 768, device='cuda')
gamma = torch.ones(768, device='cuda')
beta = torch.zeros(768, device='cuda')

# 使用 Triton 实现 LayerNorm
y = layer_norm_triton(x, gamma, beta)

# 使用 PyTorch 的 LayerNorm 作为参考
y_ref = torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta)

# 验证结果
print("Triton 实现与 PyTorch 实现是否一致:", torch.allclose(y, y_ref, atol=1e-5))