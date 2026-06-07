# Cross-Modal Steganography Toolkit

A modular, extensible toolkit for hiding information across different data modalities—text, audio, and images—using both classical and deep-learning approaches.

This project evolves through three major phases:

1. **Classical Steganography (LSB-based)**
2. **Deep Learning Steganography (CNN Encoder–Decoder)**
3. **Implicit Neural Representations (INRs)**

The goal is to provide a unified playground for research, experimentation, and demonstration of cross-modal steganographic systems.

---

## Features

### Phase 1 — Classical Steganography

Traditional bit-level embedding methods act as the baseline for evaluating more advanced models.

#### LSB Text Steganography (`lsbText.py`)

- Hide text inside images (PNG/JPG recommended)
- Extract hidden messages losslessly
- Optional zlib compression for payload reduction
- Optional XOR-based encryption for basic confidentiality
- Bit-level packing/unpacking routines for fine-grained control

#### Audio Steganography (`audioStego.py`)

- Hide WAV audio inside images using LSB manipulation
- Extract embedded audio with low reconstruction error
- Sample-accurate modifications to keep audio artifacts minimal

---

### Phase 2 — Deep Learning Steganography

Learned steganography using convolutional neural networks.

#### CNN-based Image-in-Image Steganography (`cnnStego.py`)

- Encoder embeds a **secret image** into a **cover image**
- Decoder reconstructs the hidden secret from the stego output
- Lightweight CNN architecture optimized for:
  - CPU-only machines
  - Optional CUDA acceleration when available
- PSNR-based quality analysis for:
  - Cover vs. Stego
  - Secret vs. Recovered Secret
- Designed to run on low-end laptops using small resolutions (e.g. 128×128)

---

### Phase 3 — INR-Based Steganography (`inrCM.py`)

Next-generation steganography using **Implicit Neural Representations (INRs)**, encoding information directly into the weights of a continuous neural field rather than manipulating discrete pixel grids.

#### Cross-Modal SIREN Steganography (`inrCM.py`)

- Shared **SIREN backbone** with two output heads — one for the cover, one for the secret
- Secret is encoded into a **key-shifted coordinate space** — without the correct integer key, extraction produces noise
- Supports all combinations of: `image`, `audio`, `text`, and `video` as cover or secret
- Supports `.png`, `.jpg`, `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.mp4`, `.avi`, `.txt`
- Five quality presets: `fast` / `low` / `medium` / `high` / `ultra` — auto-selected based on hardware
- Adaptive loss weighting per modality — text and audio secrets receive higher training priority
- PSNR-based quality evaluation with per-modality verdicts in `report.json` (`GOOD / OK / POOR / UNUSABLE`)
- Each run saves to a unique timestamped subfolder — no output files are ever overwritten
- Live console warning if secret PSNR falls below usable threshold after training

#### Hardware Notes

This phase is **compute-intensive**. A GPU is strongly recommended for usable results:

- On CPU, the `fast` preset (~1 min) is available to verify the pipeline works, but output quality will likely be poor
- The minimum recommended preset for real results is `low` (~15–45 min on CPU depending on machine)
- **Google Colab (free tier T4 GPU)** is the recommended environment for students without a GPU — `medium` quality runs in ~3 minutes
- Upload `inrCM.py` + your input files to Colab and run:

```python
!pip install torch librosa soundfile Pillow
!python inrCM.py --modal1 cover.png --modal2 secret.txt --mode hide --key 42 --quality medium
```

---

## Project Goals

- Provide a **unified framework** for experimenting with multiple steganographic approaches
- Enable comparison between:
  - Handcrafted LSB methods
  - CNN-based deep learning methods
  - INR-based implicit representations (cross-modal, weight-space encoding)
- Serve as a **reference implementation** for coursework, research projects, and demos
- Achieve truly **cross-modal steganography**, where different data types (image, audio, text, video) can be embedded into each other via learned INR models

---
