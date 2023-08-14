"""Implement grid sample.

Inspiration: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
"""
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import torch
from torch.nn import functional as F
import einops
import numpy as np


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def grid_sample_jax(input, coordinates, align_corners=False):
    # Convert to a consistent format (NHWC)
    input_nhwc = einops.rearrange(input, "b c h w -> b h w c")
    _, h, w, _ = input_nhwc.shape

    # Rescale the grid coordinates to fit the image dimensions
    if align_corners:
        coordinates = (coordinates + 1) * jnp.array([(h - 1) / 2, (w - 1) / 2])
    else:
        coordinates = (coordinates / 2 + 0.5) * jnp.array([h, w]) - 0.5

    # Extract y and x coordinates
    y_coords = coordinates[..., 0]
    x_coords = coordinates[..., 1]

    # Helper function to apply map_coordinates for each batch and channel
    def sample_channel(input_channel, x, y):
        return map_coordinates(input_channel, [x, y], order=0, mode="constant")

    # Use JAX's vmap to vectorize the operation over the batch and channel dimensions
    out = jax.vmap(
        jax.vmap(sample_channel, in_axes=(2, None, None)), in_axes=(0, 0, 0)
    )(input_nhwc, x_coords, y_coords)
    return out


# Test the function
B, C, H, W = 2, 2, 4, 4
x_torch = torch.arange(float(B * C * H * W)).reshape(B, C, H, W)
grid_torch = make_coord((H, W), flatten=True).unsqueeze(0).flip(-1).unsqueeze(1)
grid_torch = grid_torch.repeat(B, 1, 1, 1)  # repeat the grid for batch size

x_jax = jnp.array(x_torch.numpy())
grid_jax = jnp.array(grid_torch.numpy())

out_torch = F.grid_sample(x_torch, grid_torch, mode="nearest", align_corners=False)
out_torch[:, :, 0, :].permute(0, 2, 1)

out_jax = grid_sample_jax(x_jax, grid_jax, align_corners=False)

print("PyTorch Output:")
print(out_torch)

print("\nJAX Output:")
print(out_jax)

# Checking if outputs are close
print(
    "\nOutputs are close:", np.allclose(out_torch.detach().numpy(), out_jax, atol=1e-6)
)
