import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .ash import ASH


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) per sample for regularization.
    Standard in modern vision backbones (ConvNeXt, Swin).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, handle broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class Bottleneck(nn.Module):
    """
    ResNet-style Bottleneck with ASH integration and GroupNorm.
    Expansion factor: 4
    """
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            ash_p: int = 90,
            drop_path: float = 0.0
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups

        # 1x1 Conv (Compression)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, width)  # GN is stable for small batch sizes

        # 3x3 Conv (Spatial)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.GroupNorm(32, width)

        # ASH Injection: Sharpen features before expansion
        self.ash = ASH(percentile=ash_p)

        # 1x1 Conv (Expansion)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ash(out)  # Apply ASH here

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual Connection with Stochastic Depth
        out = out + self.drop_path(identity)
        out = self.relu(out)
        return out


class ASHEncoder(nn.Module):
    """
    High-Capacity Shared Encoder.
    Structure: ResNet-50 variant with ASH & GroupNorm.
    Output: Multi-scale features for Detection (Fine) and Recognition (Coarse).
    """

    def __init__(
            self,
            in_channels: int = 3,
            layers: List[int] = [3, 4, 6, 3],
            ash_p: int = 90,
            drop_path_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.inplanes = 64

        # Deep Stem (More robust than 7x7 conv)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inplanes = 128  # Updated after Deep Stem

        # Progressive Drop Path Rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layer(64, layers[0], stride=1, ash_p=ash_p, dpr=dpr[:layers[0]])
        self.layer2 = self._make_layer(128, layers[1], stride=2, ash_p=ash_p, dpr=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layer(256, layers[2], stride=2, ash_p=ash_p, dpr=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layer(512, layers[3], stride=2, ash_p=ash_p, dpr=dpr[sum(layers[:3]):])

        # Feature Dimensions: [256, 512, 1024, 2048]
        self.out_channels = [256, 512, 1024, 2048]
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, ash_p: int = 90,
                    dpr: List[float] = []) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride, bias=False),
                nn.GroupNorm(32, planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, ash_p=ash_p, drop_path=dpr[0]))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, ash_p=ash_p, drop_path=dpr[i]))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        # Return Deepest Latent + All scales for FPN usage
        return c4, [c1, c2, c3, c4]