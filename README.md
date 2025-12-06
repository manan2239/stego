# Cross-Modal Steganography Toolkit

A modular, extensible toolkit for hiding information across different data modalities—text, audio, and images—using both classical and deep-learning approaches.

This project evolves through three major phases:

1. **Classical Steganography (LSB-based)**
2. **Deep Learning Steganography (CNN Encoder–Decoder)**
3. **Implicit Neural Representations (INRs — upcoming)**

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

- Hide text inside WAV audio using LSB manipulation
- Extract embedded messages with low reconstruction error
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

### Phase 3 — INR-Based Steganography (Planned)

Next-generation steganography using **Implicit Neural Representations (INRs)**, enabling compact and robust encoding inside continuous neural fields rather than discrete pixel grids.

Planned capabilities:

- SIREN / NeRF-style MLP-based INRs
- Encoding information into the learned continuous function
- Cross-modal hiding (e.g., text → image INR, audio → image INR)
- Robustness against resizing, filtering, and standard image transforms
- High-capacity, function-level embedding instead of direct pixel edits

---

## Project Goals

- Provide a **unified framework** for experimenting with multiple steganographic approaches
- Enable comparison between:
  - Handcrafted LSB methods
  - CNN-based deep learning methods
  - INR-based implicit representations (future)
- Serve as a **reference implementation** for coursework, research projects, and demos
- Move towards truly **cross-modal steganography**, where different data types can be embedded into each other via learned models

---


