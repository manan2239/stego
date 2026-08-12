# Cross-Modal Steganography Toolkit

A modular, extensible toolkit for hiding information across different data modalities like text, audio, image, and video using both classical and deep-learning approaches. The project was developed as a B.Tech final-year project and evolves through three progressively advanced phases, unified under a single command-line interface and a common steganalysis engine.

\---

## Table of Contents

* [Overview](#overview)
* [Project Phases](#project-phases)
* [Architecture](#architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Steganalysis Engine](#steganalysis-engine)
* [Demo](#demo)
* [Project Structure](#project-structure)
* [Project Goals](#project-goals)
* [Hardware Notes](#hardware-notes)
* [Limitations \& Future Work](#limitations--future-work)
* [License](#license)

\---

## Overview

Steganography is the practice of concealing information within another medium such that its presence is imperceptible. This toolkit implements three generations of steganographic technique within a single, consistent codebase, allowing direct comparison of classical bit-manipulation methods against learned, neural approaches:

|Phase|Method|Core Idea|
|-|-|-|
|1|**LSB (Least Significant Bit)**|Classical bit-level embedding in image pixels|
|2|**CNN Encoder–Decoder**|Learned image-in-image embedding via convolutional networks|
|3|**INR / SIREN**|Cross-modal embedding directly into the weights of a continuous neural field|



A unified `uniStego.py` CLI dispatcher routes commands to the appropriate module, and a shared steganalysis engine (`steganalysis.py`) evaluates the imperceptibility and detectability of the output of any phase.

\---

## Project Phases

### Phase 1: Classical Steganography (LSB)

Bit-level embedding methods that serve as the baseline against which the learned models are evaluated.

**LSB Text Steganography (`lsb/lsbText.py`)**

* Hides text inside PNG/JPG images and extracts it losslessly
* Optional zlib compression to reduce payload size
* Optional XOR-based encryption for basic confidentiality
* Fine-grained bit-level packing/unpacking routines

**Audio Steganography (`lsb/audioStego.py`)**

* Hides WAV audio inside images via LSB manipulation
* Sample-accurate extraction with low reconstruction error

### Phase 2: Deep Learning Steganography (CNN)

**CNN Image-in-Image Steganography (`cnn/cnnStego.py`)**

* Encoder embeds a secret image into a cover image; decoder reconstructs the secret from the stego output
* Lightweight architecture tuned for CPU-only machines, with optional CUDA acceleration
* PSNR-based quality analysis for cover-vs-stego and secret-vs-recovered pairs
* Runs on modest hardware at low resolutions (e.g., 128×128)

### Phase 3: INR-Based Cross-Modal Steganography (`inr/inrCM.py`)

The most advanced phase: information is encoded directly into the weights of a continuous neural field (SIREN) rather than into a discrete pixel grid.

* Shared SIREN backbone with two output heads: one for the cover, one for the secret
* Secret is encoded in a **key-shifted coordinate space**; extraction without the correct integer key yields noise
* Supports every combination of `image`, `audio`, `text`, and `video` as cover or secret
* File support: `.png`, `.jpg`, `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.mp4`, `.avi`, `.txt`
* Five auto-selected quality presets: `fast` / `low` / `medium` / `high` / `ultra,` based on detected hardware
* Adaptive loss weighting per modality (text/audio secrets are prioritized during training)
* Per-modality PSNR verdicts (`GOOD` / `OK` / `POOR` / `UNUSABLE`) written to `report.json`
* Timestamped output folders per run and nothing is ever overwritten
* Live console warning if secret PSNR drops below a usable threshold

\---

## Architecture

Each phase is implemented as an independent module with a consistent input/output contract, allowing the CLI dispatcher and steganalysis engine to treat them uniformly:

```
Input (cover, secret) → Phase Module (lsb | cnn | inr) → Stego Output + report.json  
                                                              │  
                                                              ▼  
                                                  steganalysis.py (metrics + verdict)
```

Architecture and pipeline diagrams for the INR module (SVG/Mermaid) are included in the repository for detailed reference.

\---

## Installation

```
git clone https://github.com/manan2239/stego.git  
cd stego  
pip install -r requirements.txt
```

For Phase 3 (INR), on a machine without a dedicated GPU, Google Colab's free-tier T4 is recommended (see [Hardware Notes](#hardware-notes)).

\---

## Usage

### Unified CLI

```
python uniStego.py --phase lsb --mode hide --cover cover.png --secret secret.txt --output stego.png  
python uniStego.py --phase cnn --mode hide --cover cover.png --secret secret.png  
python uniStego.py --phase inr --mode hide --modal1 cover.png --modal2 secret.txt --key 42 --quality medium
```

### Phase 3 example (Colab)

```
!pip install torch librosa soundfile Pillow  
!python inrCM.py --modal1 cover.png --modal2 secret.txt --mode hide --key 42 --quality medium
```

\---

## Steganalysis Engine

`steganalysis.py` is a unified, blind steganalysis tool covering all three phases. Given a cover and stego image, it computes:

* **Imperceptibility metrics:** PSNR, SSIM, MSE, SNR, UQI, NCC, BER
* **Classical detection tests:** Chi-square, RS analysis
* **Signal analysis:** entropy, bit-plane analysis, frequency-domain noise
* A rule-based overall detection verdict

Currently supports image inputs only (LSB and CNN outputs, plus image-modality INR outputs). Audio/video/text steganalysis for the INR module is planned as a future extension.

```
python steganalysis.py --cover cover.png --stego stego.png  
python steganalysis.py --cover cover.png --stego stego.png --mode inr --secret secret.png  
python steganalysis.py --cover cover.png --stego stego.png --plot results --output report.json
```

|Flag|Description|
|-|-|
|`--cover` / `--stego`|Required input images|
|`--secret`|Optional payload image for cover-vs-secret comparison|
|`--mode`|`lsb`|
|`--output`|Save full JSON metrics report|
|`--plot`|Save visual report PNG (prefix name)|



Full pipeline breakdown: [`steganalysis\\\\\\\_architecture.md`](file:///C:/Users/Ytinifni/Desktop/steganalysis_architecture.md)

\---

## Project Structure

```
stego/  
├── lsb/                        \\# Phase 1: classical LSB methods  
│   ├── lsbText.py  
│   └── audioStego.py  
├── cnn/                        \\# Phase 2: CNN encoder-decoder  
│   └── cnnStego.py  
├── inr/                        \\# Phase 3: INR/SIREN cross-modal  
│   └── inrCM.py  
├── steganalysis/               \\# Unified steganalysis engine  
│   └── steganalysis.py  
├── uniStego.py                     \\# Unified CLI dispatcher  
├── steganalysis\\\\\\\_architecture.md  
├── LICENSE  
└── README.md
```

\---

## Project Goals

* Provide a unified framework for experimenting with multiple steganographic approaches
* Enable direct comparison between handcrafted LSB, learned CNN, and implicit neural (INR) methods
* Serve as a reference implementation for coursework, research, and demonstrations
* Achieve genuine cross-modal steganography via learned INR models

\---

## Hardware Notes

Phase 3 (INR) is compute-intensive:

* On CPU, the `fast` preset (\~1 min) verifies the pipeline runs, but output quality will likely be poor
* The minimum preset recommended for usable results is `low` (\~15–45 min on CPU)
* **Google Colab (free T4 GPU)** is recommended for students without local GPU access. `medium` quality completes in \~3 minutes

\---

## Limitations \& Future Work

* Steganalysis currently supports image-modality outputs only; audio/video/text steganalysis for INR is planned
* CNN phase is tuned for low resolutions on CPU-class hardware; scaling to higher resolutions is a future direction
* Planned: expanded benchmark suite comparing all three phases on a common dataset with a consolidated report

\---

## License

Released under the [MIT License](file:///C:/Users/Ytinifni/Desktop/LICENSE).

