import argparse
import os
import sys
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image

# Third-party
import wandb
from omegaconf import OmegaConf

# Project Imports (Assumed package structure)
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add root to path
from src.ward.models.components.encoder import ASHEncoder
from src.ward.models.ocr_net import GenerativeOCRNet
from src.ward.models.texting_net import GenerativeTextingNet
from src.ward.data.sfx_engine import SFXGenerator


# --- Utilities ---

class EMA:
    """
    Exponential Moving Average for stable generative training.
    Reference: https://arxiv.org/abs/1806.04458
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def setup_logger(save_dir: str):
    logger = logging.getLogger("WARD")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# --- Dataset ---

class WebtoonDataset(Dataset):
    """
    Hybrid Dataset: Real Webtoon Images + On-the-fly Synthetic SFX Generation.
    """

    def __init__(self, cfg, mode: str = 'train'):
        self.cfg = cfg
        self.mode = mode
        self.root = Path(cfg.data.root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load Clean Backgrounds (I^B)
        self.bg_paths = list(self.root.glob("**/*.jpg")) + list(self.root.glob("**/*.png"))

        # SFX Engine
        self.sfx_engine = SFXGenerator(cfg)

    def __len__(self):
        return len(self.bg_paths) * 100  # Virtual Epoch length

    def __getitem__(self, idx):
        # 1. Load Clean Image
        real_idx = idx % len(self.bg_paths)
        bg_path = self.bg_paths[real_idx]
        try:
            clean_img = Image.open(bg_path).convert("RGB")
        except:
            clean_img = Image.new("RGB", (640, 640), (255, 255, 255))

        # 2. Generate Synthetic Pair (I^A, I^B, Box, Text)
        stylized_img, clean_resized, bbox, text = self.sfx_engine.generate(clean_img)

        # 3. Transform
        img_A = self.transform(stylized_img)
        img_B = self.transform(clean_resized)

        # Return Dict for flexibility
        return {
            "img_stylized": img_A,  # I^A
            "img_clean": img_B,  # I^B
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "text": text
        }


# --- Loss Functions (Mockup for brevity) ---

class WARDLoss(nn.Module):
    """
    Composite Loss Module: Detection + Recognition + Generative + Attention Matching
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        # Add Perceptual Loss (VGG) here in real impl

    def forward(self, outputs, batch, mode='stage2'):
        losses = {}

        # 1. Generative Reconstruct Loss (Removal)
        if 'clean' in outputs:
            losses['loss_clean'] = self.l1(outputs['clean'],
                                           batch['img_clean']) * self.cfg.train.loss_weights.lambda_gen

        # 2. OCR Losses (Detection & Recog)
        if 'det' in outputs:
            # Placeholder for DB Loss
            losses['loss_det'] = self.mse(outputs['det']['prob'], torch.zeros_like(
                outputs['det']['prob'])) * self.cfg.train.loss_weights.lambda_det

        # 3. Texting Net Loss (DDPM Noise Pred)
        if 'noise_pred' in outputs:
            losses['loss_ddpm'] = self.mse(outputs['noise_pred'], outputs['noise_gt'])

        # 4. Attention Matching (Wasserstein/Hellinger)
        # This requires extracting attention maps from encoder

        total_loss = sum(losses.values())
        return total_loss, losses


# --- Training Logic ---

def train(args):
    # 1. Configuration Setup
    cfg = OmegaConf.load(args.config)

    # Merge argparse into config (optional overrides)
    if args.resume: cfg.train.resume_path = args.resume

    # Setup Dirs & Logging
    save_dir = Path(f"checkpoints/{cfg.project.name}/{args.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(save_dir))

    if args.wandb:
        wandb.init(project=cfg.project.name, name=args.name, config=OmegaConf.to_container(cfg))

    device = torch.device(cfg.project.device) if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Initializing WARD Framework on {device}...")

    # 2. Model Initialization
    # Shared Encoder with ASH
    encoder = ASHEncoder(
        in_channels=cfg.model.encoder.in_channels,
        ash_p=cfg.model.encoder.ash_percentile
    ).to(device)

    # Sub-networks
    ocr_net = GenerativeOCRNet(encoder, cfg).to(device)
    texting_net = GenerativeTextingNet(encoder, cfg).to(device)

    # EMA for stable generation
    ema_texting = EMA(texting_net)

    # 3. Optimizer & Scaler
    params = list(ocr_net.parameters()) + list(texting_net.parameters())
    optimizer = optim.AdamW(
        params,
        lr=cfg.train.optimizer.lr_backbone,
        weight_decay=cfg.train.optimizer.weight_decay
    )
    scaler = GradScaler()  # Mixed Precision

    # 4. Dataset
    train_dataset = WebtoonDataset(cfg, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.project.num_workers,
        pin_memory=True
    )

    criterion = WARDLoss(cfg).to(device)

    # 5. Training Loop
    start_iter = 0
    max_iters = cfg.train.stage1_iters if args.stage == 1 else cfg.train.stage2_iters

    logger.info(f"Starting Training Stage {args.stage}: {max_iters} iterations")

    ocr_net.train()
    texting_net.train()

    progress_bar = tqdm(range(start_iter, max_iters), initial=start_iter, desc=f"Stage {args.stage}")

    for i in progress_bar:
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        # Move to device
        img_clean = batch['img_clean'].to(device)  # I^B
        img_stylized = batch['img_stylized'].to(device)  # I^A

        optimizer.zero_grad()

        with autocast():
            loss_dict_log = {}

            # --- Forward Pass based on Stage ---

            # A. Removal Path (Always active in Stage 2, Optional in Stage 1)
            if args.stage == 2:
                ocr_out = ocr_net(img_stylized)
                # ocr_out keys: 'clean', 'det', 'rec'

            # B. Synthesis Path (DDPM)
            # Sample noise and timestep
            t = torch.randint(0, cfg.model.texting_net.time_steps, (img_clean.shape[0],), device=device).long()
            noise = torch.randn_like(img_stylized)
            x_noisy = noise  # Simplified: Should be q_sample(img_stylized, t, noise)

            noise_pred, _ = texting_net(x_noisy, t, img_clean)

            # --- Loss Calculation ---
            outputs = {'noise_pred': noise_pred, 'noise_gt': noise}
            if args.stage == 2:
                outputs.update(ocr_out)

            loss, loss_components = criterion(outputs, batch, mode=f'stage{args.stage}')

        # Backward & Step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_texting.update(texting_net)

        # Logging
        if i % cfg.train.print_interval == 0:
            loss_str = ", ".join([f"{k}:{v.item():.4f}" for k, v in loss_components.items()])
            progress_bar.set_postfix_str(f"L_tot: {loss.item():.4f} | {loss_str}")

            if args.wandb:
                wandb.log({"total_loss": loss.item(), **loss_components})

        # Save Checkpoint
        if i % cfg.train.save_interval == 0:
            ckpt_path = save_dir / f"ward_stage{args.stage}_iter_{i}.pth"
            torch.save({
                'iter': i,
                'encoder': encoder.state_dict(),
                'ocr_net': ocr_net.state_dict(),
                'texting_net': texting_net.state_dict(),  # Save EMA version in real deployment
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WARD: Webtoon Analysis by Recognizing and Detecting SFX")

    # Essential Arguments
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to Hydra config file')
    parser.add_argument('--name', type=str, default='default_run', help='Experiment name for logging')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Training stage (1: Synth Pretrain, 2: Joint)')

    # Flags
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (no save, small data)')

    args = parser.parse_args()

    # Debug override
    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        args.name = 'debug_run'

    train(args)