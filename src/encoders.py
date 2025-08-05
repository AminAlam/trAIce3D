import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_image_size: Tuple[int, int, int],
    ) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # self.num_point_embeddings: int = 1  # pos/neg point + 2 box corners
        # point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        # self.point_embeddings = nn.ModuleList(point_embeddings)
        # self.not_a_point_embed = nn.Embedding(1, embed_dim)
        # self.no_mask_embed = nn.Embedding(1, embed_dim)

    def _embed_points(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        # points = points + 0.5  # Shift to center of pixel.
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        return point_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        return points[0].shape[0]

    # def _get_device(self) -> torch.device:
    #     return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_embeddings = self._embed_points(points)
        return point_embeddings

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]

        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=2048, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)# (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, _l):
        # _l: length of the input
        return self.pe[:, :_l, :] # (1, _l, embed_dim)