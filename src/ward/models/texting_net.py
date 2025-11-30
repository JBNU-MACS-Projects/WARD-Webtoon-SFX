import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention optimized with torch.einsum.
    Injects global context (Conditions) into spatial features.
    """

    def __init__(self, dim: int, context_dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) - Query source
        context: (B, D) - Key/Value source
        """
        b, c, h, w = x.shape

        # Prepare Q
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)

        # Prepare K, V (Broadcasting context across spatial dim not needed here, done in attn)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        # Attention Score: Q * K^T
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # Aggregate: Attn * V
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        return self.to_out(out) + x


class DiffusionUNet(nn.Module):
    """
    DDPM U-Net with Global Context Injection via Cross-Attention.
    """

    def __init__(self, dim: int, channels: int = 3, cond_dim: int = 2048):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, 7, padding=3)

        # Sinusoidal Time Embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, time_dim)
        )

        # Down-sample Block with Time Injection
        self.down1 = self._make_res_block(dim, dim * 2, time_dim)
        self.attn1 = CrossAttention(dim * 2, cond_dim)  # Inject Style Z
        self.down2 = self._make_res_block(dim * 2, dim * 4, time_dim)

        # Bottleneck
        self.mid_block1 = self._make_res_block(dim * 4, dim * 4, time_dim)
        self.mid_attn = CrossAttention(dim * 4, cond_dim)
        self.mid_block2 = self._make_res_block(dim * 4, dim * 4, time_dim)

        # Up-sample Block
        self.up1 = self._make_res_block(dim * 8, dim * 2, time_dim)  # Concat skip
        self.attn_up1 = CrossAttention(dim * 2, cond_dim)
        self.up2 = self._make_res_block(dim * 4, dim, time_dim)

        self.final = nn.Conv2d(dim, channels, 1)

    def _make_res_block(self, in_c, out_c, time_c):
        return ResBlockWithTime(in_c, out_c, time_c)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, 1)
        t_emb = self.time_mlp(time.float().unsqueeze(-1))

        # Cond: Global Average Pooling of Z -> (B, 2048)
        cond_vec = rearrange(cond, 'b c h w -> b c (h w)').mean(dim=-1).unsqueeze(1)

        x = self.init_conv(x)

        # Down
        r1 = x
        x = self.down1(x, t_emb)  # Down to dim*2
        x = self.attn1(x, cond_vec)
        r2 = x
        x = self.down2(x, t_emb)  # Down to dim*4

        # Mid
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, cond_vec)
        x = self.mid_block2(x, t_emb)

        # Up (Using simple concat for skip connection logic)
        # Note: Actual U-Net requires proper sizing, using interpolate for simplicity here
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, r2), dim=1)
        x = self.up1(x, t_emb)
        x = self.attn_up1(x, cond_vec)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat((x, r1), dim=1)
        x = self.up2(x, t_emb)

        return self.final(x)


class ResBlockWithTime(nn.Module):
    """ ResNet Block that accepts Time Embedding """

    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_c)
        self.time_proj = nn.Linear(time_c, out_c)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        # Shift-and-Scale (AdaGN style) or just Add
        h = self.conv(x)
        h = self.norm(h)
        # Add time embedding
        scale = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return self.act(h + scale)


class GenerativeTextingNet(nn.Module):
    def __init__(self, encoder: nn.Module, cfg):
        super().__init__()
        self.encoder = encoder
        self.unet = DiffusionUNet(dim=64, cond_dim=2048)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, clean_img: torch.Tensor):
        with torch.no_grad():  # Usually condition encoder is frozen or learned jointly
            cond, _ = self.encoder(clean_img)
        return self.unet(x_t, t, cond), cond