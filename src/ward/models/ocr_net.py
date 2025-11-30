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
    """

    def __init__(self, output_size: Tuple[int, int] = (32, 100), fiducial_num: int = 20, enc_dim: int = 2048):
        super().__init__()
        self.output_size = output_size
        self.fiducial_num = fiducial_num

        # Localization Network
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

        # [Fix] Generate correct number of control points (Top & Bottom edges)
        n = fiducial_num // 2
        ctrl_pts_x = torch.linspace(-1.0, 1.0, n)
        ctrl_pts_y_top = torch.ones(n) * -1.0
        ctrl_pts_y_bottom = torch.ones(n) * 1.0

        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        ctrl_pts = torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0).flatten()

        self.loc_net[-1].bias.data.copy_(ctrl_pts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder for actual warping logic
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DBNet++) Head.
    """

    def __init__(self, in_channels: int, inner_channels: int = 256):
        super().__init__()
        self.binarize = self._build_head(in_channels, inner_channels)
        self.threshold = self._build_head(in_channels, inner_channels)

    def _build_head(self, in_c, inner_c):
        return nn.Sequential(
            nn.Conv2d(in_c, inner_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_c // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_c // 4, inner_c // 4, 2, 2),
            nn.BatchNorm2d(inner_c // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_c // 4, 1, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'prob': self.binarize(x), 'thresh': self.threshold(x)}


class PixelShuffleDecoder(nn.Module):
    """
    High-fidelity Inpainting Decoder using PixelShuffle.
    """

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
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

        # [Fix] Access config via cfg.model.ocr_net
        ocr_cfg = cfg.model.ocr_net

        self.decoder = PixelShuffleDecoder(enc_dim)
        self.det_head = DBHead(enc_dim)
        self.tps = TPS_SpatialTransformer(enc_dim=enc_dim)

        self.rec_proj = nn.Conv2d(enc_dim, 512, 1)
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        # Use corrected config path
        self.cls = nn.Linear(512, ocr_cfg.num_classes)

    def forward(self, stylized_img: torch.Tensor) -> OCRNetOutput:
        z, _ = self.encoder(stylized_img)

        clean = self.decoder(z)
        det_out = self.det_head(z)

        z_rect = self.tps(z)
        z_seq = rearrange(self.rec_proj(z_rect), 'b c h w -> b (h w) c')
        rnn_out, _ = self.lstm(z_seq)
        rec_logits = self.cls(rnn_out)

        return {'clean': clean, 'det': det_out, 'rec': rec_logits}