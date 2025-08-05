
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Type, List, Any
import numpy as np
import math
from einops import rearrange

class PatchMerging(nn.Module):
    def __init__(self, emb_size, H, W, D):
        super().__init__()
        self.H = H
        self.D = D
        self.W = W
        self.linear = nn.Linear(8*emb_size, 2*emb_size)

    def forward(self, x):
        B, L, C = x.shape
        H = self.H
        W = self.W
        D = self.D 
        x = rearrange(x, 'b (h s1 w s2 d s3) c -> b (h w d) (s1 s2 s3 c)', s1=2, s2=2, s3=2, d=D, h=H, w=W)
        x = self.linear(x)
        return x

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class UNet_ImageEncoderViT3D(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage and positional encodings."""
    def __init__(
        self,
        device,
        img_size: int = (512, 512, 32),
        patch_size: int = (16, 16, 4),
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU(),
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth

        # if self.depth % 5 != 0:
        #     raise ValueError("depth must be divisible by 5.")

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
            stride=(patch_size[0], patch_size[1], patch_size[2]),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # patch_sizes = [
        #     # (16, 32, 8),  # Large patches
        #     # (32, 16, 8),    # Medium patches
        #     (8, 8, 2)
        # ]

        # self.patch_embed = AdaptivePatchEmbedding(img_size, patch_sizes, in_chans, self.embed_dim)
        
        self.num_patches = 0
        # for patch_size in patch_sizes:
        self.num_patches = self.num_patches + img_size[0]//patch_size[0] * img_size[1]//patch_size[1] * img_size[2]//patch_size[2]
        self.pos_emb_matrix = get_sinusoidal_positional_embeddings(self.num_patches, embed_dim).to(device)
        
        self.patch_merging = nn.ModuleList()
        self.patch_merging.append(PatchMerging(embed_dim, H=16, W=16, D=4))
        self.patch_merging.append(PatchMerging(embed_dim*2, H=8, W=8, D=2))
        self.patch_merging.append(PatchMerging(embed_dim*4, H=4, W=4, D=1))


        self.patch_merged_pos_emb = [get_sinusoidal_positional_embeddings(self.num_patches//8, embed_dim*2).to(device),
                                     get_sinusoidal_positional_embeddings(self.num_patches//64, embed_dim*4).to(device),
                                     get_sinusoidal_positional_embeddings(self.num_patches//512, embed_dim*8).to(device)
        ]

        

        self.blocks = nn.ModuleList()
        self.block_indices = [(0, 1), (1, 2), (2, 3), (3, 6)]
        for i in range(4):
            for _ in range(self.block_indices[i][0], self.block_indices[i][1]):
                block = TransformerBlock(
                    dim=embed_dim*(2**i),
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    dropout_rate=dropout_rate,
                )
                self.blocks.append(block)

        print(f'--- VisionTransformer has {self.num_patches} initial patches and {len(self.blocks)} blocks')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        # print(self.pos_emb_matrix.shape)
        # x = x + self.pos_emb_matrix.repeat(x.shape[0], 1, 1).to(x.device)
        pos_emb = self.pos_emb_matrix.expand(x.shape[0], -1, -1)
        x = x + pos_emb # use of expand instead of repeat for better memory allocation
        output = []
        for i in range(4):
            for blk in self.blocks[self.block_indices[i][0]:self.block_indices[i][1]]:
                x = blk(x)

            output.append(x.contiguous())

            if i < 3:
                x = self.patch_merging[i](x)
                new_pos_emb = self.patch_merged_pos_emb[i]
                x = x + new_pos_emb.expand(x.shape[0], -1, -1)

        return output # List of tensors of shape B x H' * W' * D' x embed_dim where H'=H/patch_size[0], W'=W/patch_size[1], D'=D/patch_size[2]

class ImageEncoderViT3D(nn.Module):
    def __init__(
        self,
        img_size: int = (512, 512, 32),
        patch_size: int = (16, 16, 4),
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU(),
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size[0], patch_size[1], patch_size[2]),
            stride=(patch_size[0], patch_size[1], patch_size[2]),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.pos_emb_matrix = get_sinusoidal_positional_embeddings(self.num_patches, embed_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                dropout_rate=dropout_rate,
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_emb_matrix.repeat(x.shape[0], 1, 1).to(x.device)
        for blk in self.blocks:
            x = blk(x)
        return x

class FeedForwardNetwork(nn.Module):
    """Feed Forward Network for ViT."""
    def __init__(self, embed_dim, mlp_dim, act_layer=nn.GELU(), dropout=0.1):
        """
        Args:
            embed_dim (int): Embedding dimension.
            mlp_dim (int): Hidden layer dimension.
            act_layer (nn.Module): Activation layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_layer = act_layer
    
    def forward(self, x):
        x = self.act_layer(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU(),
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.norm2 = norm_layer(dim)

        self.ffn = FeedForwardNetwork(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = shortcut + x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = shortcut + x
        return x

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x
    

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Args:
            emb_dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
                positional parameter size.
        """
        super().__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads

        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)

        # self.attention_block = nn.MultiheadAttention(self.emb_dim, self.num_heads, dropout=dropout_rate, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, emb_dim = x.shape # B, H * W * D, emb_dim
        # qkv with shape (B, N, 3, num_heads, head_dim)

        qkv = self.qkv(x)  # Shape: (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim) # B, N , 3, num_heads, head_dim
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3, B, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            # q, k, v should be shaped as [batch, heads, seq_len, head_dim]
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_rate)
        x = x.transpose(1, 2).contiguous().reshape(B, N, self.emb_dim)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int, int] = (16, 16, 4),
        stride: Tuple[int, int, int] = (16, 16, 4),
        padding: Tuple[int, int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # B, C, H, W, D -> B, embed_dim, H', W', D' where H'=H/patch_size[0], W'=W/patch_size[1], D'=D/patch_size[2]
        x = x.flatten(2) # B, embed_dim, H', W', D' -> B, embed_dim, H' * W' * D'
        x = x.transpose(1, 2) # B, embed_dim, H' * W' * D' -> B, H' * W' * D', embed_dim
        return x
    

class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_sizes, in_channels=1, embed_dim=768):
        super().__init__()
        embed_dim = embed_dim*len(patch_sizes)
        self.patch_embeddings = nn.ModuleList()
        self.patch_sizes = patch_sizes
        
        for patch_size in patch_sizes:
            embedding = nn.Conv3d(
                in_channels, 
                embed_dim // len(patch_sizes),  # Split embedding dimension
                kernel_size=patch_size,
                stride=patch_size
            )
            self.patch_embeddings.append(embedding)

    def forward(self, x):
        embeddings = []
        for embedding in self.patch_embeddings:
            embed = embedding(x)
            # print(embed.shape)
            embeddings.append(embed.flatten(2).transpose(1, 2))
            # print(embeddings[-1].shape)
        return torch.cat(embeddings, dim=-2)  # Concatenate along embedding dimension


def get_sinusoidal_positional_embeddings(num_patches, embed_dim):
    """
    Get sinusoidal positional embeddings.
    Args:
        num_patches (int): Number of patches.
        embed_dim (int): Embedding dimension.
    Returns:
        pos_embed (Tensor): Sinusoidal positional embeddings.
    """
    position = torch.arange(num_patches).unsqueeze(1) # num_patches, 1
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim)
    )
    pos_embed = torch.zeros(num_patches, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position.float() * div_term)
    pos_embed[:, 1::2] = torch.cos(position.float() * div_term)

    return pos_embed # num_patches, embed_dim
