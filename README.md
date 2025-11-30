<div align="center">

# 🎨 WARD: Webtoon Analysis by Recognizing and Detecting SFX

[![CVPR Submission](https://img.shields.io/badge/CVPR_2026-Submission-B31B1B.svg)](https://arxiv.org/abs/2026.xxxxx)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![uv](https://img.shields.io/badge/uv-managed-purple)](https://github.com/astral-sh/uv)

**Official PyTorch Implementation of "WARD: Webtoon Analysis by Recognizing and Detecting SFX"**

[**Project Page**](#) | [**Paper**](#) | [**Demo**](#)

</div>

---

## 📖 Abstract

> **"Sound effects in Webtoons are not just text; they are visual art."**

Conventional OCR systems fail on Webtoon Sound Effects (SFX) due to extreme stylistic deformations, transparent strokes, and severe occlusions. We propose **WARD**, a novel **Bidirectional Generative OCR Framework** that reformulates SFX recognition as a dual task of **Removal ($I^A \to I^B$)** and **Synthesis ($I^B \to I^A$)**.

By leveraging a shared **ASH (Adaptive Swish)** encoder and a **DDPM-based Texting-Net**, WARD learns the underlying physical laws of SFX formation, achieving state-of-the-art performance on stylized Korean text.

## ✨ Key Features

- **🧠 ASH-Driven Shared Encoder**: Implements **Activation Shaping** to selectively amplify stroke textures while suppressing complex background noise.
- **🔄 Bidirectional Generative Modeling**:
  - **Removal Path**: Reconstructs clean backgrounds while detecting text.
  - **Synthesis Path**: Uses **DDPM (Diffusion)** to generate high-fidelity stylized SFX.
- **🎨 Infinite SFX Engine**: A physics-based stochastic rendering engine that generates infinite training data with **Elastic Deformation** and **3D Perspective Projection**.
- **⚡ Production-Ready Engineering**:
  - **TPS (Thin-Plate Spline)** for rectifying curved text.
  - **AMP (Automatic Mixed Precision)** & **EMA** training supported.
  - Managed via **`uv`** for ultra-fast dependency resolution.

---

## 🛠️ Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for modern, fast Python package management.

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/ward-ocr.git](https://github.com/your-username/ward-ocr.git)
cd ward-ocr

# 2. Initialize environment & Install dependencies
uv init
uv python pin 3.10
uv sync
````

Alternatively, using `pip`:

```bash
pip install -r requirements.txt
```

-----

## 📂 Data Preparation

WARD relies on a hybrid dataset strategy: Real Webtoon images (AI Hub) and On-the-fly Synthetic Generation.

1.  **Download Backgrounds**: Place clean webtoon panels in `data/background/`.
2.  **Fonts**: Add Korean/English `.ttf` fonts to `data/assets/fonts/`.
3.  **Corpus**: Add onomatopoeia text list to `data/assets/sfx_source/onomatopoeia.txt`.

Structure your `data/` directory as follows:

```text
data/
├── background/      # Clean images (I^B)
├── raw/             # Real labeled data (Optional for fine-tuning)
└── assets/
    ├── fonts/       # *.ttf
    └── sfx_source/  # corpus.txt
```

-----

## 🚀 Training

WARD is trained in two stages to ensure stability and generative fidelity.

### Stage 1: Synthetic Pretraining (Generative Texting-Net)

Pretrains the diffusion model using the **Infinite SFX Engine**. The model learns to synthesize stylized text on clean backgrounds.

```bash
uv run python scripts/train.py \
    --config configs/config.yaml \
    --stage 1 \
    --name pretrain_sfx \
    --wandb
```

### Stage 2: Joint Bidirectional Training

Jointly optimizes the **OCR-Net (Removal/Detection)** and **Texting-Net** with Cycle-Consistency and Attention Matching losses.

```bash
uv run python scripts/train.py \
    --config configs/config.yaml \
    --stage 2 \
    --name joint_run \
    --resume checkpoints/ward-ocr/pretrain_sfx/last.pth \
    --wandb
```

> **Note:** Training supports multi-GPU and AMP automatically. Adjust `batch_size` in `configs/config.yaml` based on your VRAM.

-----

## ⚡ Inference & Demo

Run the full pipeline (Detection → Rectification → Recognition → Removal) on your own images.

```bash
uv run python scripts/inference.py \
    --ckpt checkpoints/ward-ocr/joint_run/best.pth \
    --input assets/samples/webtoon_01.jpg \
    --output results/ \
    --device cuda
```

**Output Visualization:**
The script generates a side-by-side comparison showing the detected bounding boxes and the text-removed (clean) result.

-----

## 📊 Performance

Comparison on the **AI Hub Webtoon Dataset** (Real SFX):

| Model | Precision | Recall | H-mean |
| :--- | :---: | :---: | :---: |
| Naver Clova OCR | 14.5% | 9.4% | 24.6% |
| Google Cloud Vision | 16.2% | 5.8% | 26.5% |
| GPT-4o (VLM) | \< 10% | \< 10% | \< 10% |
| **WARD (Ours)** | **74.5%** | **67.2%** | **67.2%** |

*WARD significantly outperforms commercial OCR engines and VLMs on highly stylized text.*

-----

## 🧱 Project Structure

```text
ward-ocr/
├── configs/               # Hydra Configuration
├── src/
│   └── ward/
│       ├── models/
│       │   ├── components/
│       │   │   ├── ash.py        # Activation Shaping Layer
│       │   │   └── encoder.py    # Shared ResNet-FPN Encoder
│       │   ├── ocr_net.py        # Removal & Recognition Head (TPS+DBNet)
│       │   └── texting_net.py    # Diffusion Synthesis Head (DDPM)
│       ├── data/
│       │   └── sfx_engine.py     # Infinite SFX Construction Engine
│       └── training/
└── scripts/
    ├── train.py           # Two-stage training script
    └── inference.py       # End-to-end inference pipeline
```

-----


## Model Weight

You can download the pre-trained weight via [HuggingFace](https://huggingface.co/KyungsuLee/WARD-Webtoon-SFX)


-----


## Datasets

You can download the pre-trained weight via [HuggingFace](https://huggingface.co/datasets/KyungsuLee/WARD-Webtoon-SFX-dataset)

-----


## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

-----

<div align="center">
<sub>Implemented with ❤️ by the WARD Team.</sub>
</div>
