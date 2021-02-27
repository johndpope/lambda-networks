import torch
from torch import nn, einsum
from einops import rearrange
from typing import Optional

# helpers functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos

# lambda layer

class LambdaLayer(nn.Module):
    """Lambda layer from `"LambdaNetworks: Modeling long-range interactions without attention"
    <https://openreview.net/pdf?id=xTJEN-ggl1b>`_.
    Args:
        in_channels (int): input channels
        out_channels (int, optional): output channels
        dim_k (int): key dimension
        n (int, optional): number of input pixels
        r (int, optional): receptive field for relative positional encoding
        num_heads (int, optional): number of attention heads
        dim_u (int, optional): intra-depth dimension
    """
    def __init__(
        self,
        in_channels: int,
        *,
        dim_k: int,
        n: Optional[int] = None,
        r: Optional[int] = None,
        num_heads: int = 4,
        dim_out= None,
        dim_u: int = 1
    )-> None:
        super().__init__()
        dim_out = default(dim_out, in_channels)
        self.u = dim_u # intra-depth dimension
        self.num_heads = num_heads

        assert (dim_out % num_heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // num_heads

        # Project input and context to get queries, keys & values
        self.to_q = nn.Conv2d(in_channels, dim_k * num_heads, 1, bias = False)
        self.to_k = nn.Conv2d(in_channels, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(in_channels, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * num_heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hh, ww, u, h = *x.shape, self.u, self.num_heads

        # Project inputs & context to retrieve queries, keys and values
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Normalize queries & values
        q = self.norm_q(q)
        v = self.norm_v(v)

        # B x (num_heads * dim_k) * H * W -> B x num_heads x dim_k x (H * W)e
        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        # Normalized keys
        k = k.softmax(dim=-1)

        # Content function
        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        # Position functione
        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b h v n', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out
