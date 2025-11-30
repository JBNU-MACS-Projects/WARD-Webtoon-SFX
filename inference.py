import argparse
import os
import sys
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ward.models.components.encoder import ASHEncoder
from src.ward.models.ocr_net import GenerativeOCRNet


# TextingNet is not needed for inference unless doing style transfer demo

# --- Utils for Post Processing ---

class PostProcessor:
    """
    Handles converting raw model outputs (Heatmaps, Logits) into actionable data (Boxes, Text).
    Includes DBNet post-processing logic (Box expansion).
    """

    def __init__(self, thresh: float = 0.3, box_thresh: float = 0.5, max_candidates: int = 1000,
                 unclip_ratio: float = 1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def boxes_from_bitmap(self, pred: torch.Tensor, dest_width: int, dest_height: int) -> List[np.ndarray]:
        """
        Convert probability map to bounding boxes.
        Args:
            pred: (1, H, W) tensor, probability map
        """
        bitmap = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        scale_h = dest_height / bitmap.shape[0]
        scale_w = dest_width / bitmap.shape[1]

        # 1. Binarization
        _, binary = cv2.threshold(bitmap, self.thresh * 255, 255, cv2.THRESH_BINARY)

        # 2. Find Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for contour in contours[:self.max_candidates]:
            # Filter small noise
            if cv2.contourArea(contour) < 50:  # Minimum area
                continue

            # 3. Unclip (Box Expansion) - DBNet Logic
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))

            if points.shape[0] < 4: continue

            score = self._box_score_fast(bitmap, points)
            if score < self.box_thresh: continue

            # Apply offset (unclip)
            poly = self._unclip(points)
            if poly is None: continue

            # Scale back to original image size
            poly[:, 0] = poly[:, 0] * scale_w
            poly[:, 1] = poly[:, 1] * scale_h

            boxes.append(poly.astype(np.int32))

        return boxes

    def _box_score_fast(self, bitmap, points):
        h, w = bitmap.shape[:2]
        box = cv2.boundingRect(points)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        return cv2.mean(bitmap, mask)[0]

    def _unclip(self, box):
        # Implementation of Vatti clipping algorithm offset
        # Simplified: Use shapely or pyclipper in production.
        # Here we use a simple scaling for demonstration or OpenCV dilation logic placeholder.
        # For a truly robust implementation, install `pyclipper`.
        # import pyclipper
        # ... logic ...
        return box  # Return original for now to avoid extra dependency

    def decode_text(self, logits: torch.Tensor, vocab: List[str]) -> str:
        """
        Greedy decoding for CTC/Seq2Seq logits.
        """
        # logits: (SeqLen, NumClasses)
        pred_indices = torch.argmax(logits, dim=-1).cpu().numpy()

        # Simple CTC decoding (collapse repeats, remove blank)
        # Assuming index 0 is blank
        text = []
        prev_idx = -1
        for idx in pred_indices:
            if idx != 0 and idx != prev_idx:
                if idx < len(vocab):
                    text.append(vocab[idx])
            prev_idx = idx
        return "".join(text)


# --- Inference Engine ---

class WARDInferenceEngine:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"[WARD] Loading model from {checkpoint_path}...")
        self._load_model(checkpoint_path)
        self.post_processor = PostProcessor()

        # Mock Vocab (Korean + English + Symbols)
        # In production, load from file
        self.vocab = ['[BLANK]'] + list("가나다라마바사아자차카타파하ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?")

    def _load_model(self, ckpt_path):
        # 1. Init Architecture
        self.encoder = ASHEncoder(
            in_channels=self.cfg.model.encoder.in_channels,
            ash_p=self.cfg.model.encoder.ash_percentile
        ).to(self.device)

        self.model = GenerativeOCRNet(self.encoder, self.cfg).to(self.device)

        # 2. Load Weights
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Handle state dict keys if wrapped in DDP or compiled
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.model.load_state_dict(checkpoint['ocr_net'])

        self.model.eval()
        self.encoder.eval()

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, Any]:
        # 1. Preprocess
        original_img = Image.open(image_path).convert("RGB")
        w, h = original_img.size

        # Resize to model input size (e.g., 640x640)
        input_size = tuple(self.cfg.data.img_size)
        img_resized = original_img.resize(input_size, Image.BICUBIC)

        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float()
        img_tensor = (img_tensor / 127.5) - 1.0  # Normalize [-1, 1]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 2. Forward Pass
        outputs = self.model(img_tensor)
        # outputs: {'clean': (B,3,H,W), 'det': {'prob':...}, 'rec': ...}

        # 3. Post Process

        # A. Clean Image Restoration
        clean_tensor = outputs['clean'].squeeze(0).cpu()
        clean_tensor = (clean_tensor + 1.0) * 127.5
        clean_np = clean_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        clean_img = Image.fromarray(clean_np).resize((w, h), Image.BICUBIC)  # Resize back

        # B. Detection (Boxes)
        prob_map = outputs['det']['prob']
        boxes = self.post_processor.boxes_from_bitmap(prob_map, w, h)

        # C. Recognition (Text)
        # Note: Real WARD crops features using boxes here.
        # Since 'rec' output in this simplified code is global/seq, we just decode it.
        # In full impl, you loop through 'boxes', crop ROI, and feed to RecHead.
        raw_text = self.post_processor.decode_text(outputs['rec'].squeeze(0), self.vocab)

        return {
            "original": original_img,
            "clean": clean_img,
            "boxes": boxes,
            "text": raw_text  # This would be a list in full ROI implementation
        }


# --- Visualization ---

def visualize_result(res: Dict, save_path: str):
    """
    Creates a side-by-side comparison: Original(with Boxes) | Clean Result
    """
    orig = res['original'].copy()
    clean = res['clean'].copy()

    draw = ImageDraw.Draw(orig)
    try:
        font = ImageFont.truetype("data/fonts/NanumGothic.ttf", size=24)
    except:
        font = ImageFont.load_default()

    # Draw Boxes & Text
    for box in res['boxes']:
        # box is numpy array of shape (N, 2)
        pts = [tuple(pt) for pt in box]
        draw.polygon(pts, outline=(255, 0, 0), width=3)

        # Draw text label background
        x, y = pts[0]
        draw.text((x, y - 20), res.get('text', 'SFX'), fill=(255, 255, 0), font=font)

    # Concatenate
    w, h = orig.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(orig, (0, 0))
    combined.paste(clean, (w, 0))

    combined.save(save_path)
    print(f"Saved visualization to {save_path}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="WARD Inference Script")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory")
    parser.add_argument("--output", type=str, default="outputs/inference", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = WARDInferenceEngine(args.config, args.ckpt, args.device)

    # Get Images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

    if not image_paths:
        print("No images found.")
        return

    # Run Inference
    print(f"Starting inference on {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        try:
            result = engine.predict(str(img_path))

            # Save Clean Raw
            clean_name = f"{img_path.stem}_clean.png"
            result['clean'].save(output_dir / clean_name)

            # Save Vis
            vis_name = f"{img_path.stem}_vis.jpg"
            visualize_result(result, str(output_dir / vis_name))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()