import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypedDict, Dict, Tuple
from einops import rearrange


# Strict Output Typing
class OCRNetOutput(TypedDict):
    clean: torch.Tensor
    det: Dict[str, torch.Tensor]
    rec: torch.Tensor


class TPS_SpatialTransformer(nn.Module):
    """
    Rectifies curved/irregular text using Thin Plate Spline (TPS) transformation.
    Essential for Webtoon SFX.
    """

    def __init__(self, output_size: Tuple[int, int] = (32, 100), fiducial_num: int = 20, enc_dim: int = 2048):
        super().__init__()
        self.output_size = output_size
        self.fiducial_num = fiducial_num

        # Localization Network: Regress control points from feature map
        self.loc_net = nn.Sequential(
            nn.Conv2d(enc_dim, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, fiducial_num * 2)
        )

        # Initialize bias to identity transform
        self.loc_net[-1].weight.data.zero_()
        initial_ctrl_pts = torch.linspace(-1.0, 1.0, fiducial_num // 2)
        # Create a grid of points [[-1, -1], [-0.5, -1], ..., [1, 1]]
        ctrl_pts = torch.stack([initial_ctrl_pts, initial_ctrl_pts], dim=1).flatten()
        self.loc_net[-1].bias.data.copy_(ctrl_pts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: Full grid_sample implementation is omitted for brevity but required for production.
        # This module currently returns features directly to allow flow compilation.
        # Real implementation involves solving the linear system K * W = V
        batch_ctrl_pts = self.loc_net(x).view(-1, self.fiducial_num, 2)
        return x  # Placeholder for actual warped features


class DBHead(nn.Module):
    """
    Differentiable Binarization (DBNet++) Head.
    Predicts: Probability Map (P) and Threshold Map (T).
    """

    def __init__(self, in_channels: int, inner_channels: int = 256):
        super().__init__()
        self.binarize = self._build_head(in_channels, inner_channels)
        self.threshold = self._build_head(in_channels, inner_channels)

    def _build_head(self, in_c, inner_c):
        return nn.Sequential(
            nn.Conv2d(in_c, inner_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_c // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_c // 4, inner_c // 4, 2, 2),  # Upsample
            nn.BatchNorm2d(inner_c // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_c // 4, 1, 2, 2),  # Upsample
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'prob': self.binarize(x), 'thresh': self.threshold(x)}


class PixelShuffleDecoder(nn.Module):
    """
    High-fidelity Inpainting Decoder using PixelShuffle (Sub-pixel Convolution).
    Avoids checkerboard artifacts common in simple Upsample+Conv.
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        # 32x upsampling needed (from Latent back to Image)
        # Layer structure: Conv -> PixelShuffle -> LeakyReLU
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 1024, 3, 1, 1), nn.PixelShuffle(2), nn.GroupNorm(32, 256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.PixelShuffle(2), nn.GroupNorm(32, 128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 1, 1), nn.PixelShuffle(2), nn.GroupNorm(32, 64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 1), nn.PixelShuffle(2), nn.GroupNorm(16, 32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, out_channels, 3, 1, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenerativeOCRNet(nn.Module):
    def __init__(self, encoder: nn.Module, cfg):
        super().__init__()
        self.encoder = encoder
        enc_dim = 2048  # ResNet50 equivalent

        # 1. Removal Pathway
        self.decoder = PixelShuffleDecoder(enc_dim)

        # 2. Detection Pathway
        self.det_head = DBHead(enc_dim)

        # 3. Recognition Pathway
        self.tps = TPS_SpatialTransformer(enc_dim=enc_dim)
        self.rec_proj = nn.Conv2d(enc_dim, 512, 1)  # Reduce dim
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(512, cfg.ocr_net.num_classes)

    def forward(self, stylized_img: torch.Tensor) -> OCRNetOutput:
        # Shared Encoder
        z, _ = self.encoder(stylized_img)

        # Generative Removal
        clean = self.decoder(z)

        # Detection Map
        det_out = self.det_head(z)

        # Recognition Sequence
        z_rect = self.tps(z)  # Rectify features
        # Flatten spatial: (B, C, H, W) -> (B, Seq, C)
        z_seq = rearrange(self.rec_proj(z_rect), 'b c h w -> b (h w) c')
        rnn_out, _ = self.lstm(z_seq)
        rec_logits = self.cls(rnn_out)

        return {'clean': clean, 'det': det_out, 'rec': rec_logits}