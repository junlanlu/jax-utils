"""Implement unfold.

Inspiration: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
"""
import jax
import jax.numpy as jnp
from jax.lax import conv_general_dilated_local
import torch
from torch.nn import functional as F
import einops
import numpy as np


def unfold_jax(input, kernel_size, stride=1, padding=0, dilation=1):
    # Define dimension numbers for the operation
    dimension_numbers = ("NHWC", "IHWO", "NHWC")

    # Extract patches using the conv_general_dilated_patches function
    patches = jax.lax.conv_general_dilated_patches(
        lhs=input,
        filter_shape=kernel_size,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        lhs_dilation=(dilation, dilation),
        dimension_numbers=dimension_numbers,
    )

    # Reshape the patches to match PyTorch's nn.Unfold output
    unfolded = patches.reshape(
        patches.shape[0], patches.shape[1] * patches.shape[2], -1
    )
    unfolded = unfolded.transpose((0, 2, 1))

    return unfolded


# Test the function

x1 = torch.arange(32.0).reshape(1, 2, 4, 4)
x1_unf = F.unfold(x1, kernel_size=3, padding=1)
print(x1_unf)

x2 = einops.rearrange(x1, "b c h w -> b h w c")
x2_unf = unfold_jax(x2, kernel_size=(3, 3), padding=1)
x2_unf = torch.from_numpy(np.array(x2_unf))
print(x2_unf)
print("\nOutputs are close:", np.allclose(x1_unf.detach().numpy(), x2_unf, atol=1e-6))
