import torch.nn as nn
import torch
import math
import torch.nn.functional as F


def get_time_embedding(time_steps, temb_dim):

    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class ResBlockCustom(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Define 1x1 convolution for matching channels
        self.match_channels = nn.Conv2d(2*in_ch if up else in_ch, out_ch, kernel_size=1,
                                        bias=False) if in_ch != out_ch else nn.Identity()

        # Define convolutional layers
        self.conv_1 = nn.Sequential(
            nn.Conv2d(2*in_ch if up else in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )

        if self.time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.ReLU()
            )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x, t=None):
        # print(f"ResBlockCustom input shape: {x.shape}")

        residual = self.match_channels(x)  # Match channels if needed
        # print(f"Residual shape: {residual.shape}")

        x = self.conv_1(x)

        if self.time_emb_dim is not None:
            # Time embedding processing
            time_emb = self.time_mlp(t)
            time_emb = time_emb[(...,) + (None,) * 2]  # Expand embedding to match spatial dims
            x = x + time_emb

        x = self.conv_2(x)

        # Add residual connection
        x = x + residual
        # print(f"ResBlockCustom output shape: {x.shape}")

        return x


class ResBlockCustomAdj(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Define 1x1 convolution for matching channels
        self.match_channels = nn.Conv2d(2*in_ch if up else in_ch, out_ch, kernel_size=1,
                                        bias=False) if in_ch != out_ch else nn.Identity()

        # Define convolutional layers
        self.conv_1 = nn.Sequential(
            nn.Conv2d(2*in_ch if up else in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(out_ch),
        )

        if self.time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.LeakyReLU()
            )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(out_ch),
        )

    def forward(self, x, t=None):
        # print(f"ResBlockCustom input shape: {x.shape}")

        residual = self.match_channels(x)  # Match channels if needed
        # print(f"Residual shape: {residual.shape}")

        x = self.conv_1(x)

        if self.time_emb_dim is not None:
            # Time embedding processing
            time_emb = self.time_mlp(t)
            time_emb = time_emb[(...,) + (None,) * 2]  # Expand embedding to match spatial dims
            x = x + time_emb

        x = self.conv_2(x)

        # Add residual connection
        x = x + residual
        # print(f"ResBlockCustom output shape: {x.shape}")

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Linear transformations for Query, Key, and Value matrices
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

        # Output linear transformation
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # Assuming x is of shape [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        flattened_dim = height * width

        # Reshape to [batch_size, channels, flattened_dim] for processing
        x = x.view(batch_size, channels, flattened_dim).permute(0, 2, 1)  # [batch_size, flattened_dim, channels]

        # Calculate Query, Key, and Value matrices
        q = self.q_proj(x)  # [batch_size, flattened_dim, channels]
        k = self.k_proj(x).permute(0, 2, 1)  # Transpose for [batch_size, channels, flattened_dim]
        v = self.v_proj(x)

        # Attention scores and weights
        attention_scores = torch.bmm(q, k) / math.sqrt(channels)  # [batch_size, flattened_dim, flattened_dim]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to the values
        attended_values = torch.bmm(attention_weights, v)  # [batch_size, flattened_dim, channels]

        # Project output back to input shape
        attended_values = self.out_proj(attended_values)  # [batch_size, flattened_dim, channels]

        # Reshape back to [batch_size, channels, height, width]
        return attended_values.permute(0, 2, 1).view(batch_size, channels, height, width)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        # Ensure channels are divisible by the number of heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.head_dim = in_channels // num_heads

        # Projections for Query, Key, and Value matrices for each head
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

        # Output projection after attention aggregation
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # Assuming x is of shape [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        flattened_dim = height * width

        # Reshape to [batch_size, channels, flattened_dim] for processing
        x = x.view(batch_size, channels, flattened_dim).permute(0, 2, 1)  # [batch_size, flattened_dim, channels]

        # Project to Q, K, V matrices
        q = self.q_proj(x).view(batch_size, flattened_dim, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, flattened_dim, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, flattened_dim, self.num_heads, self.head_dim)

        # Transpose to shape [batch_size, num_heads, flattened_dim, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)  # Transpose for key matrix
        v = v.permute(0, 2, 1, 3)

        # Attention scores and weights computation
        attention_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)  # [batch, num_heads, flattened, flattened]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, v)  # [batch_size, num_heads, flattened_dim, head_dim]

        # Reshape back to [batch_size, flattened_dim, in_channels]
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, flattened_dim, channels)

        # Project to the original input shape
        attended_values = self.out_proj(attended_values)  # [batch_size, flattened_dim, channels]

        # Reshape back to [batch_size, channels, height, width]
        return attended_values.permute(0, 2, 1).view(batch_size, channels, height, width)


class UNetCustResMul3AttenAdj2(nn.Module):
    def __init__(self, image_channels=1):
        super().__init__()
        down_sample = (64, 128, 256, 512)
        up_sample = list(reversed(down_sample))
        self.time_emb_dim = time_emb_dim = 512

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial conv
        self.conv_in = nn.Conv2d(image_channels, down_sample[0], 7, padding=3)

        # Downsample path
        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResBlockCustom(down_sample[i], down_sample[i + 1], time_emb_dim),
                MultiHeadAttentionBlock(down_sample[i + 1]) if i in [2] else nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2) if i < (len(down_sample) - 2) else nn.Identity()
            ]) for i in range(len(down_sample) - 1)
        ])

        # Bottleneck attention block
        self.attention_block = MultiHeadAttentionBlock(down_sample[-1])

        # Upsample path
        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResBlockCustom(up_sample[i], up_sample[i + 1], time_emb_dim, up=True),
                MultiHeadAttentionBlock(up_sample[i + 1]) if i in [1] else nn.Identity(),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    # Two 3x3 conv as equivalent to 5x5 conv
                    nn.Conv2d(up_sample[i + 1], up_sample[i + 1], kernel_size=3, padding=1),
                    nn.Conv2d(up_sample[i + 1], up_sample[i + 1], kernel_size=3, padding=1)
                ) if i < (len(down_sample) - 2) else nn.Identity()
            ]) for i in range(len(up_sample) - 1)
        ])

        self.last = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(up_sample[-1]),
            nn.Conv2d(up_sample[-1], image_channels, 1)
        )

    def forward(self, x, t):
        # Initial conv
        x = self.conv_in(x)

        # t_emb -> B x t_emb_dim
        time_emb = get_time_embedding(torch.as_tensor(t).long(), self.time_emb_dim)
        time_emb = self.t_proj(time_emb)

        # Downsample
        down_outs = []
        for block, attn, down in self.downs:
            x = block(x, time_emb)
            x = attn(x)  # MHA in encoder
            down_outs.append(x)
            x = down(x)

        # Bottleneck attention block
        x = self.attention_block(x)

        # Upsample
        for block, attn, up in self.ups:
            residual_x = down_outs.pop()

            # Concatenate
            x = torch.cat((x, residual_x), dim=1)
            x = block(x, time_emb)
            x = attn(x)  # MHA in decoder
            x = up(x)

        # Final convolution
        return self.last(x)


# #########
# # VQVAE #
# #########

class DecoderVQVAE(nn.Module):
    def __init__(self, image_channels=1, z_channels=4, up_channels=(256, 128, 64), up_sample=(True, True)):
        super().__init__()
        self.up_channels = up_channels
        self.up_sample = up_sample
        self.z_channels = z_channels

        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(self.z_channels, self.up_channels[0], kernel_size=3, padding=1)

        self.ups = nn.ModuleList([])
        for i in range(len(self.up_channels) - 1):
            self.ups.append(
                nn.ModuleList([
                    ResBlockCustom(self.up_channels[i], self.up_channels[i + 1]),
                    # nn.ConvTranspose2d(self.up_channels[i + 1], self.up_channels[i + 1], 4, 2, 1)
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(self.up_channels[i + 1], self.up_channels[i + 1], kernel_size=3, padding=1),
                    nn.Conv2d(self.up_channels[i + 1], self.up_channels[i + 1], kernel_size=3, padding=1)
                ])
            )

        # Attention block at the bottleneck
        self.attention_block = AttentionBlock(self.up_channels[0])

        self.conv_out = nn.Conv2d(self.up_channels[-1], image_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.post_quant_conv(x)
        x = self.conv_in(x)

        # attention block
        x = self.attention_block(x)

        for block, up, conv1, conv2 in self.ups:
            x = block(x)
            x = up(x)
            x = conv1(x)
            x = conv2(x)

        x = self.conv_out(x)
        return x


class EncoderVQVAE(nn.Module):
    def __init__(self, image_channels=1, z_channels=4, down_channels=(64, 128, 256), down_sample=(True, True),
                 codebook_size=40):
        super().__init__()
        self.down_channels = down_channels
        self.down_sample = down_sample
        self.z_channels = z_channels
        self.codebook_size = codebook_size

        self.conv_in = nn.Conv2d(image_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                nn.ModuleList([
                    ResBlockCustom(self.down_channels[i], self.down_channels[i + 1]),
                    nn.Conv2d(self.down_channels[i + 1], self.down_channels[i + 1], 4, 2, 1)
                ])
            )

        self.conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

    def quantize(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def forward(self, x):
        x = self.conv_in(x)
        for block, down in self.downs:
            x = block(x)
            x = down(x)

        x = self.conv_out(x)
        x = self.pre_quant_conv(x)
        out, quant_losses, _ = self.quantize(x)

        return out, quant_losses


class VQVAE(nn.Module):
    def __init__(self, image_channels=1, codebook_size=40):
        super().__init__()

        self.compress_ratio = 4
        self.encoder = EncoderVQVAE(image_channels, codebook_size=codebook_size)
        self.decoder = DecoderVQVAE(image_channels)

    def forward(self, x):
        z, quant_losses = self.encoder(x)
        out = self.decoder(z)

        return out, z, quant_losses


class DecoderVQVAEAdj(nn.Module):
    def __init__(self, image_channels=1, z_channels=4, up_channels=(256, 128, 64), up_sample=(True, True)):
        super().__init__()
        self.up_channels = up_channels
        self.up_sample = up_sample
        self.z_channels = z_channels

        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(self.z_channels, self.up_channels[0], kernel_size=3, padding=1)

        self.ups = nn.ModuleList([])
        for i in range(len(self.up_channels) - 1):
            self.ups.append(
                nn.ModuleList([
                    ResBlockCustomAdj(self.up_channels[i], self.up_channels[i + 1]),
                    # nn.ConvTranspose2d(self.up_channels[i + 1], self.up_channels[i + 1], 4, 2, 1),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    # nn.PixelShuffle(2),
                    # Two 3x3 conv as equivalent to 5x5 conv
                    nn.Conv2d(self.up_channels[i + 1], self.up_channels[i + 1], kernel_size=3, padding=1),
                    nn.Conv2d(self.up_channels[i + 1], self.up_channels[i + 1], kernel_size=3, padding=1)
                ])
            )

        # Attention block at the bottleneck
        # self.attention_block = AttentionBlock(self.up_channels[0])
        self.attention_block = MultiHeadAttentionBlock(self.up_channels[0], num_heads=4)

        # batch norm
        self.last_norm = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm2d(self.up_channels[-1])
        )

        self.conv_out = nn.Conv2d(self.up_channels[-1], image_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.post_quant_conv(x)
        x = self.conv_in(x)

        # attention block
        # x = self.attention_block(x)

        for block, up, conv1, conv2 in self.ups:
            x = block(x)
            x = up(x)
            x = conv1(x)  # Two 3x3 conv as equivalent to 5x5 conv
            x = conv2(x)

        # Attention block
        # x = self.attention_block(x)

        x = self.last_norm(x),
        if isinstance(x, tuple):
            x = x[0]  # Unpack if necessary

        x = self.conv_out(x)
        return x


class EncoderVQVAEAdj(nn.Module):
    def __init__(self, image_channels=1, z_channels=4, down_channels=(64, 128, 256),
                 down_sample=(True, True), codebook_size=512):
        super().__init__()
        self.down_channels = down_channels
        self.down_sample = down_sample
        self.z_channels = z_channels
        self.codebook_size = codebook_size

        self.conv_in = nn.Conv2d(image_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                nn.ModuleList([
                    ResBlockCustomAdj(self.down_channels[i], self.down_channels[i + 1]),
                    nn.Conv2d(self.down_channels[i + 1], self.down_channels[i + 1], 4, 2, 1)
                ])
            )

        self.conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

    def quantize(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def forward(self, x):
        x = self.conv_in(x)
        for block, down in self.downs:
            x = block(x)
            x = down(x)

        x = self.conv_out(x)
        x = self.pre_quant_conv(x)
        out, quant_losses, _ = self.quantize(x)

        return out, quant_losses


class VQVAE512ADJ2(VQVAE):
    def __init__(self, image_channels=1, codebook_size=512):
        super().__init__(image_channels=image_channels)

        self.compress_ratio = 8
        self.encoder = EncoderVQVAEAdj(image_channels, z_channels=4, down_channels=(32, 64, 128, 256),
                                       down_sample=(True, True, True), codebook_size=codebook_size)
        self.decoder = DecoderVQVAEAdj(image_channels, z_channels=4, up_channels=(256, 128, 64, 32),
                                       up_sample=(True, True, True))

    def forward(self, x):
        z, quant_losses = self.encoder(x)
        out = self.decoder(z)

        return out, z, quant_losses
