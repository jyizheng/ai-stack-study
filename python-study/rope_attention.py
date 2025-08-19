import torch
import torch.nn as nn
from typing import Tuple

def precompute_rope_angles(dim: int, seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes the cosine and sine values for the rotary position embeddings.
    
    Args:
        dim (int): The dimension of the embeddings. Should be an even number.
        seq_len (int): The maximum sequence length.
        theta (float): The base for the frequency calculations.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the precomputed 
                                         cosine and sine tensors of shape (seq_len, dim // 2).
    """
    # Create the frequency "theta" vector based on the dimension
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    
    # Calculate the outer product to get a tensor of shape (seq_len, dim // 2)
    freqs = torch.outer(t, freqs).to(torch.float32)
    
    # Calculate the cosine and sine parts
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies the rotary position embeddings to the input tensor.
    
    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, seq_len, num_heads, head_dim).
        cos (torch.Tensor): The precomputed cosine tensor.
        sin (torch.Tensor): The precomputed sine tensor.
        
    Returns:
        torch.Tensor: The tensor with rotary embeddings applied.
    """
    # Reshape the tensor to separate the real and imaginary parts
    x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
    x_rotated = x_rotated.reshape_as(x).contiguous()
    
    # Expand cos and sin to match the tensor shape for broadcasting
    cos_expanded = cos.unsqueeze(0).unsqueeze(2)
    sin_expanded = sin.unsqueeze(0).unsqueeze(2)
    
    # Apply the rotation
    return x * cos_expanded + x_rotated * sin_expanded


class RotaryAttention(nn.Module):
    """
    A conceptual self-attention block with Rotary Position Embeddings (ROPE).
    """
    def __init__(self, d_model: int, num_heads: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Ensure the dimension is divisible by 2 for ROPE
        assert self.head_dim % 2 == 0, "Head dimension must be an even number for ROPE"
        
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection layer
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Precompute the ROPE angles
        self.cos, self.sin = precompute_rope_angles(self.head_dim, seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Optional attention mask.
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply ROPE to Q and K
        q = apply_rope(q, self.cos[:seq_len], self.sin[:seq_len])
        k = apply_rope(k, self.cos[:seq_len], self.sin[:seq_len])
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to V and reshape
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final output projection
        return self.out_proj(output)

# Example Usage
d_model = 512
num_heads = 8
seq_len = 128
batch_size = 2

# Instantiate the attention block
rope_attention = RotaryAttention(d_model=d_model, num_heads=num_heads, seq_len=seq_len)

# Create a random input tensor
input_tensor = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output = rope_attention(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")


