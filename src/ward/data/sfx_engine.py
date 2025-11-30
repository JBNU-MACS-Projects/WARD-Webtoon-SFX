import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Any


class SFXGenerator:
    """
    Advanced Procedural SFX Synthesis Engine.

    Generates 'Infinite' training data by applying physics-based deformations:
    1. Mesh-based Elastic Distortion (Sinusoidal warp)
    2. 3D Perspective Projection (Homography)
    3. Poisson Blending / Alpha compositing
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.canvas_size = tuple(cfg.sfx_engine.canvas_size)
        self.corpus = ["POW", "BAM", "WHOOSH", "CRASH"]  # Placeholder

    def _generate_mesh_warp(self, img: np.ndarray, magnitude: float = 10.0, freq_x: float = 150.0,
                            freq_y: float = 120.0) -> np.ndarray:
        """
        Applies non-linear elastic deformation using sinusoidal grid perturbation.
        Simulates organic paper bending or hand-drawn stroke irregularity.
        """
        H, W = img.shape[:2]
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # Perturbation fields: dx, dy
        # Using broadcasting for efficient frequency modulation
        dx = magnitude * np.sin(2 * np.pi * y / freq_y)
        dy = magnitude * np.cos(2 * np.pi * x / freq_x)

        # Add perturbation to original coordinates
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Remap pixels
        return cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def _apply_homography(self, img: np.ndarray, factor: float = 0.3) -> np.ndarray:
        """
        Simulates 3D camera angles via randomized Homography matrix.
        """
        H, W = img.shape[:2]
        src_pts = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

        # Randomized corner perturbations (Scale-invariant)
        perturb = np.random.uniform(-factor, factor, (4, 2)) * np.array([W, H])
        dst_pts = src_pts + perturb.astype(np.float32)

        # Calculate Homography Matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        return cv2.warpPerspective(
            img, M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    def generate(self, clean_img: Image.Image) -> Tuple[Image.Image, Image.Image, List[int], str]:
        """
        Pipeline: Render Text -> Elastic Warp -> Perspective Warp -> Composite
        """
        # 1. Init
        W, H = self.canvas_size
        clean_img = clean_img.resize((W, H))

        # 2. Render Basic Text (Stub)
        txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        # ... Drawing logic (Draw.text) goes here ...

        # 3. Convert to Numpy for Advanced Warps
        cv_txt = np.array(txt_layer)

        # 4. Apply Physics-based Deformations
        cv_txt = self._generate_mesh_warp(cv_txt, magnitude=np.random.uniform(5, 15))
        cv_txt = self._apply_homography(cv_txt, factor=np.random.uniform(0.1, 0.3))

        # 5. Composite back to PIL
        txt_layer_distorted = Image.fromarray(cv_txt)
        stylized_img = Image.alpha_composite(clean_img.convert("RGBA"), txt_layer_distorted).convert("RGB")

        return stylized_img, clean_img, [0, 0, 10, 10], "GeneratedText"