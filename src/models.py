import torch
import torch.nn as nn
import numpy as np
import VisTrans

import math

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

class CellSomaSegmentationModel(nn.Module):
    def __init__(self):
        super(CellSomaSegmentationModel, self).__init__()
        self.dropout_prob = 0.3
        self.num_features = 16
        
        # Encoder
        self.down_1 = nn.Sequential(
            # input size 1*128x128x16
            nn.Conv3d(1, self.num_features, kernel_size=(8,8,2), padding=(3,3,0), stride=(2,2,2)),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 8*64x64x8
        )
        self.down_2 = nn.Sequential(
            # input size 8*64x64x8
            nn.Conv3d(self.num_features, 2*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(2*self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 16*32x32x8
        )
        self.down_3 = nn.Sequential(
            # input size 16*32x32x8
            nn.Conv3d(2*self.num_features, 4*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(4*self.num_features),
            nn.ReLU(),
            # output size 32*16x16x8
        )

        self.double_conv_1 = nn.Sequential(
            # input size 32*16x16x8
            nn.Conv3d(4*self.num_features, 4*self.num_features, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            nn.Conv3d(4*self.num_features, 4*self.num_features, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 32*16x16x8
        )

        # Decoder
        self.up_1 = nn.Sequential(
            # input size 64*16x16x8
            nn.ConvTranspose3d(8*self.num_features, 4*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(4*self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 32*32x32x8
        )
        self.up_2 = nn.Sequential(
            # input size 48*32x32x8
            nn.ConvTranspose3d(6*self.num_features, 2*self.num_features, kernel_size=(8,8,3), padding=(3,3,1), stride=(2,2,1)),
            nn.BatchNorm3d(2*self.num_features),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_prob),
            # output size 16*64x64x8
        )
        self.up_3 = nn.Sequential(
            # input size 24*64x64x8
            nn.ConvTranspose3d(3*self.num_features, 1, kernel_size=(8,8,4), padding=(3,3,1), stride=(2,2,2)),
            nn.BatchNorm3d(1),
            nn.Hardtanh(min_val=0, max_val=1),
            # output size 1*128x128x16
        )
    
    def forward(self, x):
        # x shape is 1*64x64x16
        x1 = self.down_1(x)
        # x1 shape is 8*32x32x8
        x2 = self.down_2(x1)
        # x2 shape is 16*16x16x8
        x3 = self.down_3(x2)
        # x3 shape is 32*8x8x8
        x4 = self.double_conv_1(x3)
        # x4 shape is 32*8x8x8
        x5 = torch.cat([x4, x3], dim=1)
        # x5 shape is 64*8x8x8
        x6 = self.up_1(x5)
        # x6 shape is 32*16x16x8
        x7 = torch.cat([x6, x2], dim=1)
        # x7 shape is 48*16x16x8
        x8 = self.up_2(x7)
        # x8 shape is 16*32x32x8
        x9 = torch.cat([x8, x1], dim=1)
        # x9 shape is 24*32x32x16
        x10 = self.up_3(x9)
        # x10 shape is 1*64x64x16

        return x10

class PromptEncoding(nn.Module):
    def __init__(self, embed_dim=2048):
        super(PromptEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        self.conv1 = nn.Conv3d(1, 1, kernel_size=(11, 11, 3), padding=(5, 5, 1), stride=(4, 4, 4)) # 1*128*128*16 -> 1*32*32*4
        self.linear1 = nn.Linear(32*32*4, embed_dim) # 1*128*128*128 -> 1*2048
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: B*1*512*512*32
        x = self.conv1(x)
        # flatten x
        x = x.view(x.size(0), -1)        
        x = self.linear1(x)
        x = self.relu(x).unsqueeze(1)
        # x: B*1*2048
        return x

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads: int):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
class Attention_different_emb_sizes(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim_q: int,
        embedding_dim_k: int,
        embedding_dim_v: int,
        internal_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.embedding_dim_q = embedding_dim_q
        self.embedding_dim_k = embedding_dim_k
        self.embedding_dim_v = embedding_dim_v
        self.internal_dim = internal_dim
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."


        self.q_proj = nn.Linear(embedding_dim_q, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim_k, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim_v, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim_v)

    def _separate_heads(self, x, num_heads: int):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v):
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation = nn.GELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )


    def forward(
        self, queries, keys, query_pe, key_pe
    ):
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

class MLPBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), dropout_prob=0.1, num_conv_layers=3):
        super(UNetResidualBlock, self).__init__()

        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)
        self.num_conv_layers = num_conv_layers
        self.adjust = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        layers = []
        for i in range(self.num_conv_layers):
            if i == 0:
                no_in_channels = in_channels
            else:
                no_in_channels = out_channels
            
            layers.append(nn.Sequential(
                                nn.Conv3d(no_in_channels, out_channels, kernel_size=3, padding=1),
                                nn.BatchNorm3d(out_channels),
                                self.activation,
                                nn.Dropout3d(dropout_prob)
                                )
                        )
        
        self.conv_blocks = nn.Sequential(*layers)
        

    def forward(self, x):
        residual = self.adjust(x)
        out = self.conv_blocks(x)
        out += residual
        out = self.activation(out)
        out = self.dropout(out)
        return out

class U_Net_encoder(nn.Module):
    def __init__(self, dropout_prob=0.1, activation=nn.GELU()):
        super().__init__()

        self.dropout_prob = dropout_prob
        self.activation = activation

        self.double_conv_block_1 = UNetResidualBlock(1, 4, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=4)
        self.down_sample_1 = self.make_down_sample_block(4)

        self.double_conv_block_2 = UNetResidualBlock(4, 16, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=4)
        self.down_sample_2 = self.make_down_sample_block(16)

        self.double_conv_block_3 = UNetResidualBlock(16, 64, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=4)
        self.down_sample_3 = self.make_down_sample_block(64)

        self.double_conv_block_4 = UNetResidualBlock(64, 256, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=4)
        self.down_sample_4 = self.make_down_sample_block(256)

        self.double_conv_block_5 = UNetResidualBlock(256, 1024, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=4)
    
    def make_down_sample_block(self, no_channels):
        return nn.Sequential(
            nn.Conv3d(no_channels, no_channels, kernel_size=(3,3,1), stride=(2,2,2), padding=(1,1,0)),
            nn.BatchNorm3d(no_channels),
            self.activation,
        )

    def forward(self, x):
        # x in shape B, 1, H, W, D
        x1 = self.double_conv_block_1(x)
        # x1 in shape B, 32, H, W, D
        x2 = self.down_sample_1(x1)
        x2 = self.double_conv_block_2(x2)
        # x2 in shape B, 64, H/2, W/2, D/2
        x3 = self.down_sample_2(x2)
        x3 = self.double_conv_block_3(x3)
        # x3 in shape B, 128, H/4, W/4, D/4
        x4 = self.down_sample_3(x3)
        x4 = self.double_conv_block_4(x4)
        # x4 in shape B, 256, H/8, W/8, D/8
        x5 = self.down_sample_4(x4)
        x5 = self.double_conv_block_5(x5)
        # x5 in shape B, 512, H/16, W/16, D/16

        return x1, x2, x3, x4, x5

class U_Net_decoder(nn.Module):
    def __init__(self, dropout_prob=0.2, activation=nn.GELU(), skip_connections=True, num_conv_layers=4, num_patches=512, up_sampling_dim=64, emb_dim=128):
        super().__init__()

        # assert up_sampling_dim %2 == 0, 'Upsampling dimension should be a multiple of 2'
        assert up_sampling_dim>=32, 'Upsampling dimension should be greater than 32'

        self.dropout_prob = dropout_prob
        self.activation = activation
        self.skip_connections = skip_connections
        self.num_conv_layers = num_conv_layers
        self.up_sampling_dim = up_sampling_dim


        self.up_sample_4 = UpSamplingBlock(emb_dim*8, emb_dim*4, upsample_factor=2, dropout_prob=self.dropout_prob)

        self.double_conv_block_3 = UNetResidualBlock(emb_dim*8, emb_dim*8, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=num_conv_layers)
        self.up_sample_3 = UpSamplingBlock(emb_dim*8, emb_dim*2, upsample_factor=2, dropout_prob=self.dropout_prob)
        
        self.double_conv_block_2 = UNetResidualBlock(emb_dim*4, emb_dim*4, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=num_conv_layers)
        self.up_sample_2 = UpSamplingBlock(emb_dim*4, emb_dim, upsample_factor=2, dropout_prob=self.dropout_prob)

        self.double_conv_block_1 = UNetResidualBlock(emb_dim*2, emb_dim, activation=self.activation, dropout_prob=self.dropout_prob, num_conv_layers=num_conv_layers)
        self.up_sample_1 = UpSamplingBlock(emb_dim, emb_dim//2, upsample_factor=2, dropout_prob=self.dropout_prob)

        self.conv_final = nn.Sequential(
            nn.ConvTranspose3d(emb_dim//2, emb_dim//4, kernel_size=(2,2,1), stride=(2,2,1), padding=0),
            nn.BatchNorm3d(emb_dim//4),
            self.activation,
            nn.Dropout3d(self.dropout_prob),
            nn.Conv3d(emb_dim//4, emb_dim//8, kernel_size=(3,3,3), stride=(1,1,1), padding=1),
            nn.BatchNorm3d(emb_dim//8),
            self.activation,
            nn.Dropout3d(self.dropout_prob),
            nn.Conv3d(emb_dim//8, emb_dim//16, kernel_size=(3,3,3), stride=(1,1,1), padding=1),
            nn.BatchNorm3d(emb_dim//16),
            self.activation,
            nn.Dropout3d(self.dropout_prob),
            nn.ConvTranspose3d(emb_dim//16, emb_dim//32, kernel_size=(2,2,1), stride=(2,2,1), padding=0),
            nn.BatchNorm3d(emb_dim//32),
            self.activation,
            nn.Dropout3d(self.dropout_prob),
            nn.Conv3d(emb_dim//32, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=1),
        )
        
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2, x3, x4):
        # x1 in shape B, 256, 32, 32, 8
        # x2 in shape B, 512, 16, 16, 4
        # x3 in shape B, 1024, 8, 8, 2
        # x4 in shape B, 2048, 4, 4, 1

        x4 = self.up_sample_4(x4)
        if self.skip_connections:
            x5 = torch.cat([x4, x3], dim=1)
        else:
            x5 = torch.cat([x4, x4], dim=1)
        x5 = self.double_conv_block_3(x5)
        # x6 in shape B, 256, H/8, W/8, D/8

        x6 = self.up_sample_3(x5)
        if self.skip_connections:
            x6 = torch.cat([x6, x2], dim=1)
        else:
            x6 = torch.cat([x6, x5], dim=1)
        x6 = self.double_conv_block_2(x6)

        x7 = self.up_sample_2(x6)
        if self.skip_connections:
            x7 = torch.cat([x7, x1], dim=1)
        else:
            x7 = torch.cat([x7, x6], dim=1)
        x8 = self.double_conv_block_1(x7)
        
        x9 = self.up_sample_1(x8)
        
        # x9 in shape B, 32, H, W, D

        x10 = self.conv_final(x9)
        # x10 in shape B, 1, H, W, D

        # no sigmoid if using BCEWithLogitsLoss
        # x10 = self.sigmoid(x10)

        return x10



class U_Net_decoder_TransformerBased(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, num_patches, depth=4, dropout_prob=0.1, activation=nn.GELU(), skip_connections=True):
        super().__init__()

        self.dropout_prob = dropout_prob
        self.activation = activation

        print(int(mlp_ratio*embed_dim))

        self.image_emb_pe = get_sinusoidal_positional_embeddings(num_patches, embed_dim)

        self.cross_attention_x1 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(mlp_ratio*embed_dim))
        self.cross_attention_x2 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(mlp_ratio*embed_dim))
        self.cross_attention_x3 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(mlp_ratio*embed_dim))
        self.cross_attention_x4 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(mlp_ratio*embed_dim))
        self.cross_attention_x5 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(mlp_ratio*embed_dim))

        self.x1_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_prob,
        )
            self.x1_block.append(block)

        self.x2_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_prob,
        )
            self.x2_block.append(block)

        self.x3_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_prob,
        )
            self.x3_block.append(block)

        self.x4_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_prob,
        )
            self.x4_block.append(block)

        self.x5_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_prob,
        )
            self.x5_block.append(block)

        self.x1_block = nn.Sequential(*self.x1_block)
        self.x2_block = nn.Sequential(*self.x2_block)
        self.x3_block = nn.Sequential(*self.x3_block)
        self.x4_block = nn.Sequential(*self.x4_block)
        self.x5_block = nn.Sequential(*self.x5_block)

        self.up_sample_final = UpSamplingBlockCustomDim(2, 2, upsample_factor=(2,2,2))
        self.conv_final = nn.Conv3d(2, 1, kernel_size=(1,1,1), stride=(1,1,1))
        
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2, x3, x4, x5, x6):
        # x1 in shape no_pathes*emb_dim
        # x2 in shape no_pathes*emb_dim
        # x3 in shape no_pathes*emb_dim
        # x4 in shape no_pathes*emb_dim
        # x5 in shape no_pathes*emb_dim
        pos_emb = self.image_emb_pe.to(x5.device)

        img_pos_emb = self.image_emb_pe.expand(x1.shape[0], -1, -1).to(x5.device)

        _, x7 = self.cross_attention_x5(x5, x6, pos_emb, pos_emb)
        x7 = x7+img_pos_emb
        x7 = self.x5_block(x7)

        _, x8 = self.cross_attention_x4(x4, x7, pos_emb, pos_emb)
        x8 = x8+img_pos_emb
        x8 = self.x4_block(x8)

        _, x9 = self.cross_attention_x3(x3, x8, pos_emb, pos_emb)
        x9 = x9+img_pos_emb
        x9 = self.x3_block(x9)

        _, x10 = self.cross_attention_x2(x2, x9, pos_emb, pos_emb)
        x10 = x10+img_pos_emb
        x10 = self.x2_block(x10)

        _, x11 = self.cross_attention_x1(x1, x10, pos_emb, pos_emb)
        x11 = x11+img_pos_emb
        x11 = self.x1_block(x11)

        x11 = x11.view(x11.shape[0], 2, 128, 128, 8)
        x11 = self.up_sample_final(x11)
        x11 = self.conv_final(x11)

        # no sigmoid if using BCEWithLogitsLoss
        # x10 = self.sigmoid(x10)

        return x11


class TwoWayAttentionBlock_torchAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation = nn.ReLU,
        drop_out_p: float = 0.1,
    ) -> None:
        
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=drop_out_p, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads, dropout=drop_out_p, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads, dropout=drop_out_p, batch_first=True)


    def forward(
        self, queries, keys, query_pe, key_pe
    ):
        q = queries + query_pe
        attn_out, _ = self.self_attn(q, q, queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_image_to_token(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys
    
class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.GELU(), dropout_prob=0.2, upsample_factor=2):
        super().__init__()

        # if upsample_factor%2 != 0:
        #     raise ValueError('Upsample factor should be a multiple of 2')

        self.activation = activation
        self.dropout_prob = dropout_prob
        self.up_sample = self.make_up_sample_block(in_channels, 
                                                   out_channels, 
                                                   kernel_size=(upsample_factor,upsample_factor,upsample_factor), 
                                                   stride=(upsample_factor,upsample_factor,upsample_factor), 
                                                   padding=(0,0,0))
    
    def make_up_sample_block(self, in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            self.activation,
            nn.Dropout3d(self.dropout_prob),
        )
    
    def forward(self, x):
        return self.up_sample(x)

class UpSamplingBlockCustomDim(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.GELU(), dropout_prob=0.1, upsample_factor=(2,2,1)):
        super().__init__()

        # if upsample_factor%2 != 0:
        #     raise ValueError('Upsample factor should be a multiple of 2')

        self.activation = activation
        self.dropout_prob = dropout_prob
        self.up_sample = self.make_up_sample_block(in_channels, 
                                                   out_channels, 
                                                   kernel_size=upsample_factor, 
                                                   stride=upsample_factor, 
                                                   padding=(0,0,0))
    
    def make_up_sample_block(self, in_channels, out_channels, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            self.activation,
            nn.Dropout3d(self.dropout_prob),
        )
    
    def forward(self, x):
        return self.up_sample(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dropout_prob=0.1, activation=nn.GELU()):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(in_channels),
            activation,
            nn.Dropout3d(dropout_prob),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            activation,
            nn.Dropout3d(dropout_prob),
        )

    def forward(self, x):
        return self.block(x)

class SkipConnectionBlock(nn.Module):
    def __init__(self, input_img_size=(128, 128, 16), num_patches=512, linear_proj_dim=256, up_sampling_dim=64, emb_dim=64):
        super().__init__()

        self.emb_dim = emb_dim
    
        self.num_patches = num_patches
        self.linear_proj_dim = linear_proj_dim
        self.up_sampling_dim = up_sampling_dim

        self.input_img_size = input_img_size

    def forward(self, x1, x2, x3, x4):

        x1 = x1.view(x1.shape[0], self.emb_dim, 32, 32, 8) # should be 8192
        x2 = x2.view(x2.shape[0], self.emb_dim*2, 16, 16, 4) # stays as 1024
        x3 = x3.view(x3.shape[0], self.emb_dim*4, 8, 8, 2) # should be 128
        x4 = x4.view(x4.shape[0], self.emb_dim*8, 4, 4, 1) # should be 16

        return x1, x2, x3, x4
    
    def print_debug_info(self):
        print(f"SkipConnectionBlock Debug Info:")
        print(f"  num_patches: {self.num_patches}")
        print(f"  linear_proj_dim: {self.linear_proj_dim}")
        print(f"  up_sampling_dim: {self.up_sampling_dim}")
        print(f"  8x upsampling output dim: {int(self.up_sampling_dim/4)}")
        print(f"  16x upsampling output dim: {int(self.up_sampling_dim/8)}")



class SkipConnectionBlockTransformerBased(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads, mlp_ratio, depth = 4, dropout_rate=0.1):
        super().__init__()
        self.pos_emb_matrix = VisTrans.get_sinusoidal_positional_embeddings(num_patches, embed_dim)

        self.bottle_neck_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_rate,
        )
            self.bottle_neck_block.append(block)
        self.bottle_neck_block = nn.Sequential(*self.bottle_neck_block)

    def forward(self, x1, x2, x3, x4, x5):
        x6 = self.bottle_neck_block(x5)
        return x1, x2, x3, x4, x5, x6
    
class SkipConnectionBlockTransformerBasedWithCrossAttention(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads, mlp_ratio, depth = 4, dropout_rate=0.1):
        super().__init__()
        self.pos_emb_matrix = VisTrans.get_sinusoidal_positional_embeddings(num_patches, embed_dim)

        self.cross_attention_1 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(embed_dim*mlp_ratio))
        self.cross_attention_2 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(embed_dim*mlp_ratio))
        self.cross_attention_3 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(embed_dim*mlp_ratio))
        self.cross_attention_4 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(embed_dim*mlp_ratio))
        self.cross_attention_5 = TwoWayAttentionBlock_torchAttention(embed_dim, num_heads, int(embed_dim*mlp_ratio))

        self.bottle_neck_block = nn.ModuleList()
        for _ in range(depth):
            block = VisTrans.TransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU(),
            dropout_rate=dropout_rate,
        )
            self.bottle_neck_block.append(block)

        self.bottle_neck_block = nn.Sequential(*self.bottle_neck_block)

    def forward(self, ecnoded_prompt, x1, x2, x3, x4, x5, prompt_pe, image_emb_pe):
        _, x5 = self.cross_attention_1(ecnoded_prompt, x5, prompt_pe, image_emb_pe)
        _, x4 = self.cross_attention_2(ecnoded_prompt, x4, prompt_pe, image_emb_pe)
        _, x3 = self.cross_attention_3(ecnoded_prompt, x3, prompt_pe, image_emb_pe)
        _, x2 = self.cross_attention_4(ecnoded_prompt, x2, prompt_pe, image_emb_pe)
        _, x1 = self.cross_attention_5(ecnoded_prompt, x1, prompt_pe, image_emb_pe)
        x6 = self.bottle_neck_block(x5)
        return x1, x2, x3, x4, x5, x6


class SkipConnectionBlockWithCrossAttention(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, mlp_dim=128, input_img_size=(128, 128, 16), num_patches=512, linear_proj_dim=128, up_sampling_dim=64):
        super().__init__()
        self.num_patches = num_patches
        self.linear_proj_dim = linear_proj_dim
        self.up_sampling_dim = up_sampling_dim
        self.embedding_dim = embedding_dim
        print(f'skip connection cross attention emb dim {embedding_dim}, num_heads {num_heads}, mlp_dim {mlp_dim}')
        self.cross_attention_1 = TwoWayAttentionBlock_torchAttention(embedding_dim, num_heads, mlp_dim)
        self.cross_attention_2 = TwoWayAttentionBlock_torchAttention(embedding_dim*2, num_heads, mlp_dim)
        self.cross_attention_3 = TwoWayAttentionBlock_torchAttention(embedding_dim*4, num_heads, mlp_dim)
        self.cross_attention_4 = TwoWayAttentionBlock_torchAttention(embedding_dim*8, num_heads, mlp_dim)

        self.input_img_size = input_img_size

    def forward(self, ecnoded_prompt, x1, x2, x3, x4, prompt_pe, image_emb_pe):
        B, _, _ = ecnoded_prompt.shape
        
        x1 = x1 + self.cross_attention_1(ecnoded_prompt, x1, prompt_pe, image_emb_pe)[1]
        x2 = x2 + self.cross_attention_2(ecnoded_prompt.repeat(1, 1, 2), x2, prompt_pe.repeat(1, 2), image_emb_pe[0:x2.shape[1], :].repeat(1, 2))[1]
        x3 = x3 + self.cross_attention_3(ecnoded_prompt.repeat(1, 1, 4), x3, prompt_pe.repeat(1, 4), image_emb_pe[0:x3.shape[1], :].repeat(1, 4))[1]
        x4 = x4 + self.cross_attention_4(ecnoded_prompt.repeat(1, 1, 8), x4, prompt_pe.repeat(1, 8), image_emb_pe[0:x4.shape[1], :].repeat(1, 8))[1]

        x1 = x1.view(x1.shape[0], self.embedding_dim, 32, 32, 8) # should be 8192
        x2 = x2.view(x2.shape[0], self.embedding_dim*2, 16, 16, 4) # stays as 1024
        x3 = x3.view(x3.shape[0], self.embedding_dim*4, 8, 8, 2) # should be 128
        x4 = x4.view(x4.shape[0], self.embedding_dim*8, 4, 4, 1) # should be 16

        return x1, x2, x3, x4

class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.GELU(), dropout_prob=0.1, depth_factor=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_conv_layers = max(2, int(4 * depth_factor))  # Adaptive depth based on level
        self.conv_blocks = nn.ModuleList()
        
        # First convolution
        self.conv_blocks.append(self._make_conv_block(in_channels, out_channels, dropout_prob))
        
        # Middle convolutions
        for _ in range(num_conv_layers - 2):
            self.conv_blocks.append(self._make_conv_block(out_channels, out_channels, dropout_prob))
            
        # Last convolution
        self.conv_blocks.append(self._make_conv_block(out_channels, out_channels, dropout_prob))
        
        # Residual connection
        if in_channels != out_channels:
            self.adjust = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        self.dropout = nn.Dropout3d(dropout_prob)
        self.activation = activation

    def _make_conv_block(self, in_ch, out_ch, dropout_prob):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout_prob)
        )

    def forward(self, x):
        if self.in_channels != self.out_channels:
            residual = self.adjust(x)
        else:
            residual = x
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        return self.activation(x + residual)

class EnhancedSkipConnectionBlock(nn.Module):
    def __init__(self, input_img_size=(128, 128, 16), num_patches=512, linear_proj_dim=512, up_sampling_dim=64, conv_refinement=True):
        super().__init__()
        assert up_sampling_dim % 2 == 0, 'Upsampling dimension should be a multiple of 2'
        assert up_sampling_dim >= 32, 'Upsampling dimension should be greater than 32'
        
        self.num_patches = num_patches
        self.linear_proj_dim = linear_proj_dim
        self.up_sampling_dim = up_sampling_dim
        self.conv_refinement = conv_refinement

        self.up_sampling_blocks = nn.ModuleList([
            UpSamplingBlock(64, 2, upsample_factor=8), # x1 64*16*16*2 -> 2*128*128*16
            UpSamplingBlock(64, 8, upsample_factor=4), # x2 64*16*16*2 -> 8*64*64*8
            UpSamplingBlock(64, 32, upsample_factor=2), # x3 64*16*16*2 -> 32*32*32*4
            UpSamplingBlock(64, 128, upsample_factor=1) # x4 64*16*16*2 -> 128*16*16*2
        ])

        self.input_img_size = input_img_size

    def forward(self, x1, x2, x3, x4):

        B = x4.shape[0]
        x1 = x1.view(B, 64, 16, 16, 2)
        x2 = x2.view(B, 64, 16, 16, 2)
        x3 = x3.view(B, 64, 16, 16, 2)
        x4 = x4.view(B, 64, 16, 16, 2)
        # Apply upsampling
        features = []
        for i, (x, up_block) in enumerate(zip([x1, x2, x3, x4], self.up_sampling_blocks)):
            feat = up_block(x)
            features.append(feat)

        return features[0], features[1], features[2], features[3]

class ImprovedUNetDecoder(nn.Module):
    def __init__(self, dropout_prob=0.1, activation=nn.GELU(), skip_connections=True, num_patches=512, up_sampling_dim=64):
        super().__init__()
        
        self.skip_connections = skip_connections
        
        # Create decoder blocks with adaptive depth
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(64, 64, depth_factor=1.0), # for x4
            self._make_decoder_block(16, 16, depth_factor=0.75), # for x3
            self._make_decoder_block(4, 4, depth_factor=0.5), # for x2
        ])

        self.up_sample_blocks = nn.ModuleList([
            UpSamplingBlock(128, 32, upsample_factor=2), # x4 128*16*16*2
            UpSamplingBlock(64, 8, upsample_factor=2), # x3 32*32*32*4
            UpSamplingBlock(16, 2, upsample_factor=2) # x2 8*64*64*8
        ])
        
        # Final convolutions with progressive channel reduction
        self.final_convs = nn.Sequential(
            nn.Conv3d(4, 2, 3, padding=1),
            nn.BatchNorm3d(2),
            activation,
            nn.Dropout3d(dropout_prob),
            nn.Conv3d(2, 1, 3, padding=1),
            nn.BatchNorm3d(1),
            activation,
            nn.Dropout3d(dropout_prob),
            nn.Conv3d(1, 1, 1)
        )

    def _make_decoder_block(self, in_channels, out_channels, depth_factor):
        return nn.Sequential(
            AdaptiveResidualBlock(in_channels,
                                out_channels,
                                depth_factor=depth_factor)
        )

    def forward(self, x1, x2, x3, x4):
        # x4 128*16*16*2
        # x3 32*32*32*4
        # x2 8*64*64*8
        # x1 2*128*128*16

        x4 = self.up_sample_blocks[0](x4)
        x = torch.cat([x3, x4], dim=1)
        x = self.decoder_blocks[0](x)

        x = self.up_sample_blocks[1](x)
        x = torch.cat([x2, x], dim=1)
        x = self.decoder_blocks[1](x)

        x = self.up_sample_blocks[2](x)
        x = torch.cat([x1, x], dim=1)
        x = self.decoder_blocks[2](x)

        x = self.final_convs(x)

        return x

class CellBranchSegmentationModel_UNet_noSkipConn_VisionTransformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x5 = self.encoder(x)
        x5 = x5.view(x5.shape[0], x5.shape[1], 8, 8, 1)
        x10 = self.decoder(None, None, None, None, x5)
        return x10

class CellBranchSegmentationModel_UNet_VisionTransformer(nn.Module):
    def __init__(self, encoder, decoder, freeze_encoder=False):
        super().__init__()
        self.input_img_size = encoder.img_size
        self.encoder = encoder
        self.decoder = decoder

        self.up_sampling_block_2times = UpSamplingBlock(512, 256, upsample_factor=2)
        self.up_sampling_block_4times = UpSamplingBlock(512, 128, upsample_factor=4)
        self.up_sampling_block_8times = UpSamplingBlock(512, 64, upsample_factor=8)
        self.up_sampling_block_16times = UpSamplingBlock(512, 32, upsample_factor=16)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        [x1, x2, x3, x4, x5] = self.encoder(x)

        x5 = x5.view(x5.shape[0], x5.shape[1], int(self.input_img_size[0]/16), int(self.input_img_size[1]/16), int(self.input_img_size[2]/16))
        x4 = x4.view(x4.shape[0], x4.shape[1], int(self.input_img_size[0]/16), int(self.input_img_size[1]/16), int(self.input_img_size[2]/16))
        x3 = x3.view(x3.shape[0], x3.shape[1], int(self.input_img_size[0]/16), int(self.input_img_size[1]/16), int(self.input_img_size[2]/16))
        x2 = x2.view(x2.shape[0], x2.shape[1], int(self.input_img_size[0]/16), int(self.input_img_size[1]/16), int(self.input_img_size[2]/16))
        x1 = x1.view(x1.shape[0], x1.shape[1], int(self.input_img_size[0]/16), int(self.input_img_size[1]/16), int(self.input_img_size[2]/16))

        # x4 in shape B, 256, H/8, W/8, D/8
        x4 = self.up_sampling_block_2times(x4)
        # x3 in shape B, 128, H/4, W/4, D/4
        x3 = self.up_sampling_block_4times(x3)
        # x2 in shape B, 64, H/2, W/2, D/2
        x2 = self.up_sampling_block_8times(x2)
        # x1 in shape B, 32, H, W, D
        x1 = self.up_sampling_block_16times(x1)
        
        x10 = self.decoder(x1, x2, x3, x4, x5)
        return x10

class UNet(nn.Module):
    ''' UNet model '''
    def __init__(self, encoder, decoder, skip_connection_block, freeze_encoder=False, freeze_skip_connection_block=False, freeze_decoder=False):
        super().__init__()
        self.input_img_size = encoder.img_size
        self.encoder = encoder
        self.decoder = decoder
        self.skip_connection_block = skip_connection_block

        self.image_encoder_embed_dim = encoder.embed_dim
        self.image_encoder_num_patches = encoder.num_patches

        self.image_emb_pe = get_sinusoidal_positional_embeddings(self.image_encoder_num_patches, self.image_encoder_embed_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('Encoder frozen')

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print('Decoder frozen')

        if freeze_skip_connection_block:
            for param in self.skip_connection_block.parameters():
                param.requires_grad = False
            print('Skip connection block frozen')

    def forward(self, x):
        [x1, x2, x3, x4] = self.encoder(x)

        x1, x2, x3, x4 = self.skip_connection_block(x1, x2, x3, x4)

        x = self.decoder(x1, x2, x3, x4)
        return x

class CubeHead(nn.Module):
    def __init__(self, no_heads=4, embed_dim=64, mlp_dim=128):
        super().__init__()
        self.no_heads = no_heads
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, no_heads)


class UNetWithCubeHead(nn.Module):
    ''' UNet model with a cube head. The cube head is used to predict if the cell is cut by the cube sides or not. '''
    def __init__(self, encoder, decoder, skip_connection_block, freeze_encoder=False, freeze_skip_connection_block=False, freeze_decoder=False):
        super().__init__()
        self.input_img_size = encoder.img_size
        self.encoder = encoder
        self.decoder = decoder
        self.skip_connection_block = skip_connection_block

        self.image_encoder_embed_dim = encoder.embed_dim
        self.image_encoder_num_patches = encoder.num_patches

        self.image_emb_pe = get_sinusoidal_positional_embeddings(self.image_encoder_num_patches, self.image_encoder_embed_dim)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('Encoder frozen')

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            print('Decoder frozen')

        if freeze_skip_connection_block:
            for param in self.skip_connection_block.parameters():
                param.requires_grad = False
            print('Skip connection block frozen')

    def forward(self, x):
        [x1, x2, x3, x4, x5] = self.encoder(x)

        x1, x2, x3, x4, x5 = self.skip_connection_block(x1, x2, x3, x4, x5)

        x10 = self.decoder(x1, x2, x3, x4, x5)
        return x10
    
class UNet_PromptDriven(nn.Module):
    ''' UNet model with prompt driven attention mechanism. The skip connection block is used to combine the prompt embedding with the encoder output using attention mechanism.'''
    def __init__(self, encoder, decoder, prompt_encoder, skip_connection_block, freeze_encoder=False, freeze_prompt_encoder=False, freeze_skip_connection_block=False, freeze_decoder=False):
        super().__init__()
        self.input_img_size = encoder.img_size
        self.encoder = encoder
        self.decoder = decoder
        self.prompt_encoder = prompt_encoder
        self.skip_connection_block = skip_connection_block

        self.prompt_encoder_embed_dim = prompt_encoder.embed_dim
        self.image_encoder_embed_dim = encoder.embed_dim
        self.image_encoder_num_patches = encoder.num_patches

        self.prompt_pe = get_sinusoidal_positional_embeddings(1, self.prompt_encoder_embed_dim)
        self.image_emb_pe = get_sinusoidal_positional_embeddings(self.image_encoder_num_patches, self.image_encoder_embed_dim)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        if freeze_skip_connection_block:
            for param in self.skip_connection_block.parameters():
                param.requires_grad = False

    def forward(self, x, prompt):
        [x1, x2, x3, x4] = self.encoder(x)
        ecnoded_prompt = self.prompt_encoder(prompt)

        self.image_emb_pe = self.image_emb_pe.to(x.device)
        self.prompt_pe = self.prompt_pe.to(x.device)

        x1, x2, x3, x4 = self.skip_connection_block(ecnoded_prompt, x1, x2, x3, x4, self.prompt_pe, self.image_emb_pe)

        x10 = self.decoder(x1, x2, x3, x4)
        return x10