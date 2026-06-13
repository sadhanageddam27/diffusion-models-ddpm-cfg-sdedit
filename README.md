# Diffusion Models — DDPM, CFG, SDEdit, and Visual Anagrams

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![DeepFloyd](https://img.shields.io/badge/DeepFloyd-IF-purple)

Three-part deep learning project exploring diffusion models from the ground
up — forward/reverse DDPM processes, Classifier-Free Guidance for controlled
generation, SDEdit for image-to-image translation, and Visual Anagrams
that embed two different semantic meanings into a single generated image.

## Live Project Report
Full results with images and analysis:
** https://sadhanageddam27.github.io/diffusion-models-ddpm-cfg-sdedit//**

---

## What This Project Covers

### Part 1 — Forward and Reverse Diffusion Processes
Implemented the core DDPM mechanism using the DeepFloyd IF Stage-1 UNet.

| Method | Quality | Notes |
|--------|---------|-------|
| Gaussian blur | Poor | Removes noise but destroys edges |
| One-step UNet | Medium | Sharper than blur, fails at high noise |
| Iterative DDPM | Best | Multi-step reverse chain recovers full structure |

### Part 2 — Generation, CFG, and SDEdit

**Classifier-Free Guidance (CFG)** at scale γ = 7:


**SDEdit** tested on tower, flower, and cat images:
- Low noise → stays close to original
- Medium noise → subtle stylistic changes  
- High noise → creative reinterpretation, global structure preserved

### Part 3 — Visual Anagrams
Single images that show one concept upright and a different concept
when flipped 180°. Two noise predictions blended at every denoising step.

| Upright | Flipped |
|---------|---------|
| Campfire | Old Man |
| Waterfalls | Skull |
| Dog | Man |

Results at 64×64 (Stage-1) upsampled to 256×256 (Stage-2).

---

## Note on API Keys
This notebook uses the HuggingFace API. Never hardcode tokens — use:
```python
import os
hf_token = os.environ.get("HF_TOKEN")
```

---

## Setup and Usage
```bash
git clone https://github.com/sadhanageddam27/diffusion-models-ddpm-cfg-sdedit.git
pip install torch diffusers transformers accelerate jupyter matplotlib
jupyter notebook
```
Recommended: run on Google Colab with GPU runtime.

---

## Tech Stack
Python · PyTorch · HuggingFace Diffusers · DeepFloyd IF · Jupyter · Matplotlib

## Topics
`diffusion-models` `ddpm` `classifier-free-guidance` `sdedit`
`deep-learning` `pytorch` `computer-vision` `python`
