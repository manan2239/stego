# Steganalysis Pipeline — Architecture & Technical Breakdown
**Group 02 · IUST Kashmir · Cross-Modal Steganography**  
`steganalysis.py` — Unified toolkit covering LSB (Sem 6), CNN (Sem 7), INR (Sem 8)

---

## Overview

The script is a **blind steganalysis pipeline** — it takes a cover image and a suspected stego image (it does not need the embedding algorithm) and runs a battery of statistical, structural, and frequency-domain tests to measure imperceptibility, estimate payload, and produce a detection verdict. It is mode-aware: you can tell it which phase produced the stego image, or let it auto-detect.

---

## Stage 0 — Entry & Input Validation

**Entry point:** `main()` → `run_analysis()`

```
CLI args
  --cover     (required)
  --stego     (required)
  --secret    (optional — enables cover vs secret comparison)
  --mode      lsb | cnn | inr | auto
  --output    save JSON report
  --plot      save PNG visual report
  --quiet     suppress rich console
```

Both images are loaded via Pillow → converted to RGB → cast to `float32 [0, 255]`. If the stego image dimensions differ from the cover, it is Lanczos-resampled to match. The secret image (if provided) gets the same treatment.

---

## Stage 1 — Mode Detection

**Function:** `detect_mode(cover, stego)`

When `--mode auto` (default), the pipeline heuristically decides which steganography method likely produced the stego image, because different methods leave different statistical fingerprints:

| Signal | Interpretation |
|---|---|
| >90% of changed pixels differ by exactly ±1 | LSB — bit flips are always ±1 |
| LSB change rate > 10% of total pixels | Confirms LSB (payload is spread wide) |
| Noise std < 2.0 (smooth additive noise) | CNN — encoder adds small uniform noise |
| Larger multi-value diffs | INR — coordinate-based residuals vary more |

The detected mode is stored in the metrics dict and passed to the verdict engine so mode-specific tests are applied correctly.

---

## Stage 2 — Imperceptibility Metrics

**Functions:** `psnr()`, `ssim()`, `mse()`, `snr()`, `uqi()`, `ncc()`, `bit_error_rate()`

This is the core quality measurement layer. All seven metrics run on every analysis regardless of mode. Each answers a slightly different question about the distortion introduced by embedding.

### PSNR — Peak Signal-to-Noise Ratio
```
PSNR = 10 · log10(255² / MSE)
```
Measures the ratio between the maximum possible signal power and the noise power. The standard benchmark in steganography papers. Implemented via scikit-image when available; falls back to the direct formula.

- > 40 dB → good imperceptibility (target for all three phases)
- 30–40 dB → moderate, perceptible under scrutiny
- < 30 dB → visible distortion

### SSIM — Structural Similarity Index
```
SSIM(x,y) = [luminance] × [contrast] × [structure]
```
Captures what PSNR misses: the human visual system cares more about structural changes than raw pixel error. A stego image can have low MSE but still look wrong if edges are smeared. Falls back to a simplified per-channel calculation if scikit-image is absent.

- > 0.99 → good
- 0.95–0.99 → acceptable
- < 0.95 → structural distortion visible

### MSE — Mean Squared Error
```
MSE = mean((cover − stego)²)
```
The raw squared pixel error averaged across all pixels and channels. Used as the basis for PSNR. Penalises large errors quadratically, so outlier pixels (where the embedding is strong) dominate.

### SNR — Signal-to-Noise Ratio
```
SNR = 10 · log10(mean(cover²) / mean((cover − stego)²))
```
Unlike PSNR, SNR uses the actual signal power of the cover (not the fixed maximum of 255²). This makes it sensitive to dark images where PSNR would look artificially high.

### UQI — Universal Quality Index
```
Q = [4 · σ_xy · μ_x · μ_y] / [(σ_x² + σ_y²)(μ_x² + μ_y²)]
```
Wang & Bovik (2002). Decomposes distortion into three independent components: loss of correlation (structure), luminance distortion, and contrast distortion. Range [−1, 1]; 1 = identical. More comprehensive than PSNR alone; useful for comparing CNN and INR outputs where distortion can be non-uniform.

### NCC — Normalized Cross-Correlation
```
NCC = Σ(cover · stego) / Σ(cover²)
```
Widely used in watermarking literature. Robust to uniform brightness shifts. Value of 1.0 means the stego image is a perfect scaled copy of the cover. Deviations indicate structural change.

### BER — Bit Error Rate
```
BER = differing_bits / total_bits
       (computed via XOR at the 8-bit level)
```
Counts what fraction of all bits in the image have been flipped. Particularly meaningful for LSB steganalysis — a full 1-bit LSB embed changes ~50% of LSBs ≈ BER of 1/8 = 0.125. Values near 0 for CNN/INR mean the embedding is numerically small; high BER for LSB is expected and normal.

---

## Stage 3 — Pixel Difference Statistics

**Function:** `pixel_difference_stats(cover, stego)`

Works on the raw difference array `|cover − stego|` and the signed noise `cover − stego`. Complements the aggregate metrics above with distributional information.

| Statistic | What it tells you |
|---|---|
| Max absolute diff | Worst-case pixel change (high for INR, usually 1 for LSB) |
| Mean absolute diff | Average embedding strength |
| Median absolute diff | Robust central tendency (unaffected by outliers) |
| Changed pixels count/% | How spatially widespread the embedding is |
| Pixels diff by ±1 | Diagnostic for LSB — almost all changes are ±1 |
| Pixels diff by >1 | Diagnostic for CNN/INR — larger multi-value changes |
| Noise skewness | Symmetry of the noise distribution (LSB ≈ 0; CNN can skew) |
| Noise kurtosis | Tailedness — Gaussian noise has kurtosis 3; LSB noise has high kurtosis (sparse impulses) |

---

## Stage 4 — Shannon Entropy Analysis

**Function:** `entropy_analysis(cover, stego)`

```
H(X) = −Σ p(x) · log₂(p(x))
```

Computed separately per channel (R, G, B) for both the cover and stego image, using the 256-bin pixel value histogram as the probability distribution.

Steganographic embedding tends to increase entropy:
- LSB embedding with a random payload pushes the pixel histogram toward uniform → higher entropy
- CNN-based embedding introduces smooth correlated noise → entropy change is smaller
- INR embedding may redistribute pixel values in spatially coherent ways → entropy change is subtle

The delta `Δ = stego_entropy − cover_entropy` is the key output. A delta > 0.001 bits triggers a yellow flag in the verdict engine.

---

## Stage 5 — LSB Bit-Plane Analysis

**Function:** `lsb_plane_analysis(cover, stego)`

Extracts all 8 bit-planes (bit 0 = LSB to bit 7 = MSB) for all three channels, for both images. For each plane, computes:

- **cover_ones_ratio** — fraction of 1s in the cover's bit-plane (baseline randomness)
- **stego_ones_ratio** — fraction of 1s in the stego's bit-plane
- **flip_rate** — fraction of pixels where the bit changed between cover and stego

For pure 1-bit LSB steganography with a random payload:
- Bit 0 flip rate will be ~0.5 (half the LSBs are flipped)
- Bits 1–7 flip rates will be ~0 (untouched)

For 2-bit LSB: bits 0 and 1 both show high flip rates. For CNN/INR: all flip rates are low, but spread across bits 0–3 depending on embedding strength. This makes the bit-plane profile a useful mode signature, and is visualised as the "Bit-Plane Flip Rates" line chart in the plot.

---

## Stage 6 — Chi-Square Histogram Test

**Function:** `histogram_analysis(cover, stego)`

The classical LSB steganalysis attack (Westfeld & Pfitzmann, 1999). Based on the observation that LSB embedding with a random payload forces adjacent value pairs (0,1), (2,3), (4,5), ..., (254,255) to become equally frequent, because flipping an LSB turns value 2k into 2k+1 and vice versa.

For each channel:

1. Build the 256-bin pixel value histogram of the stego image.
2. For each pair (2k, 2k+1), treat the expected count as `(count_2k + count_{2k+1}) / 2`.
3. Compute the chi-square statistic against this expected distribution.
4. Compute the p-value from the chi-square CDF with df = number of pairs.

**Interpretation:** p-value < 0.05 → the histogram pair distribution is statistically inconsistent with a natural image → LSB embedding suspected. The test also computes the histogram correlation between cover and stego as a secondary signal.

This is one of the oldest and most reliable attacks against naive sequential LSB embedding. It is less effective against adaptive/spread LSB, which is why the RS analysis complements it.

---

## Stage 7 — RS Steganalysis

**Function:** `rs_analysis(img_arr)`

Fridrich, Goljan & Du (2001). The most principled classical LSB detector. Rather than using histograms, it measures the statistical smoothness of pixel groups and how flipping operations change that smoothness.

**Groups:** Consecutive 4-pixel horizontal runs.

**Discriminant function `f(G)`:** Sum of absolute differences between adjacent pixels in the group. Natural images are smooth → low f. Random payloads make images rougher → higher f.

**Two flipping operations:**
- `+F`: flip LSB of every other pixel in the group
- `−F`: flip all LSBs, then apply `+F`

**Classification:** For each group G:
- Apply `+F` → get F(G). If `f(F(G)) > f(G)`, group is **Regular (R)**; if less, **Singular (S)**.
- Apply `−F` → get R⁻, S⁻ analogously.

**The key property:** In a natural image (no stego), `R ≈ R⁻` and `S ≈ S⁻`. LSB embedding breaks this symmetry in a predictable way.

**Payload estimation:** Solve the quadratic:
```
2(R⁻ − R) · p² + (R − R⁻ + S⁻ − S) · p + (S⁻ − S) = 0
```
The solution `p ∈ [0, 1]` is the estimated fraction of pixels carrying a payload. ~0% = clean image; ~50% = half the LSBs embedded; ~100% = all LSBs used.

**Performance note:** RS analysis is O(H×W) and slow in pure Python for large images. For images > 512×512, the script automatically crops to a 512×512 center region.

---

## Stage 8 — Frequency Domain Analysis

**Function:** `frequency_domain_analysis(cover, stego)`

Computes the 2D FFT of the **noise residual** (`cover − stego`) for each channel, then analyses the spatial frequency distribution of the embedding noise.

```
noise_fft = |fftshift(fft2(cover_channel − stego_channel))|
```

The FFT magnitude is split into two radial bands using an Euclidean distance mask from the DC centre:
- **Low frequency** (inner quarter radius): coarse spatial structure
- **High frequency** (everything else): fine texture, edges, noise

**Why this matters per mode:**

| Mode | Frequency signature of noise |
|---|---|
| LSB | Random white-noise payload → energy spread uniformly → high-freq dominance |
| CNN | Encoder learns smooth noise → lower high-freq ratio |
| INR | Coordinate-network residuals → can produce structured patterns → variable |

Also reports noise std, max, and min per channel for reference. CNN stego typically has noise std < 2.0; LSB has noise std ≈ 0.5 (sparse ±1 changes); INR varies.

---

## Stage 9 — Optional: Secret vs Cover Comparison

When `--secret` is provided, runs a reduced metric set (PSNR, SSIM, MSE, mean diff) comparing the **cover against the secret payload** rather than against the stego. This answers the question: how different is the thing being hidden from the carrier? A high-entropy secret (e.g. a compressed image or noise image) is harder to hide imperceptibly than a smooth low-entropy secret. The scatter plot of cover G vs secret G in the visual report visualises this.

---

## Stage 10 — Verdict Engine

**Function:** `steganalysis_verdict(metrics, mode)`

Aggregates all computed signals into a structured verdict. Rule-based — no ML.

**Confidence levels:**

| Confidence | Trigger condition |
|---|---|
| HIGH | Chi-square detected in ≥2/3 channels OR RS payload estimate > 5% |
| MEDIUM | Neither HIGH condition met, but ≥3 flags raised overall |
| LOW | Fewer than 3 flags |

**Flag logic (evaluated in order):**

1. PSNR bucket: < 30 → poor, 30–40 → moderate, > 40 → good imperceptibility
2. SSIM < 0.95 → structural distortion flag
3. BER > 0.001 → bit-level change flag
4. Chi-square: counts channels where `p < 0.05` → LSB detection flag (mode = lsb/auto)
5. RS payload: mean across channels > 5% → payload fraction flag (mode = lsb/auto)
6. Frequency: any channel with high-freq noise ratio > 0.7 → deep stego artifact flag

`stego_suspected = True` when 2 or more flags are raised (any single metric is insufficient; convergence of evidence is required).

---

## Stage 11 — Visual Report (optional)

**Function:** `generate_plots(cover, stego, metrics, prefix, secret=None)`

Produces a 3×5 (or 3×6 with secret) matplotlib grid saved as `{prefix}_steganalysis.png`.

**Row 0 — Image visualisations:**
- Cover image
- Stego image
- Amplified difference (`|cover − stego| × 20`) — makes subtle embedding visible
- Cover LSB plane (G channel) — visualises natural LSB structure
- Stego LSB plane (G channel) — random-looking = payload embedded
- Secret image (if provided)

**Row 1 — Statistical visualisations:**
- R, G, B histograms: cover (coloured) vs stego (white overlay)
- Noise distribution histogram — shape reveals embedding type
- Bit-plane flip rates (0–7) per channel — signature plot of embedding depth

**Row 2 — Analysis visualisations:**
- FFT of noise (G channel, log scale, magma colormap) — frequency fingerprint
- Normalised imperceptibility bar chart (PSNR/50, SSIM, UQI, NCC, SNR/50)
- RS payload % per channel bar chart
- Entropy comparison (cover vs stego per channel)
- Verdict summary panel (confidence, stego suspected, flags)
- Cover vs secret scatter plot (G channel, if secret provided)

---

## Output Schema (JSON)

```json
{
  "mode": "lsb | cnn | inr",
  "cover_path": "...",
  "stego_path": "...",
  "image_shape": [H, W, 3],
  "image_size_px": N,

  "imperceptibility": {
    "psnr_db": float,
    "ssim": float,
    "mse": float,
    "snr_db": float,
    "uqi": float,
    "ncc": float,
    "bit_error_rate": float
  },

  "capacity": {
    "lsb_1bit": { "bits": N, "bytes": N, "bpp": 3 },
    ...
    "deep_1bpp": { "bits": N, "bytes": N, "bpp": 1 },
    "deep_3bpp": { "bits": N, "bytes": N, "bpp": 3 }
  },

  "pixel_differences": {
    "max_absolute_diff": float,
    "mean_absolute_diff": float,
    "changed_pixels_count": int,
    "changed_pixels_pct": float,
    "pixels_diff_by_1": int,
    "pixels_diff_gt_1": int,
    "noise_skewness": float,
    "noise_kurtosis": float
  },

  "entropy": {
    "R": { "cover_entropy_bits": float, "stego_entropy_bits": float, "entropy_delta": float },
    "G": { ... },
    "B": { ... }
  },

  "lsb_planes": {
    "R": {
      "bit_0": { "cover_ones_ratio": float, "stego_ones_ratio": float, "flip_rate": float },
      ...
      "bit_7": { ... }
    },
    "G": { ... },
    "B": { ... }
  },

  "histogram_chi2": {
    "R": { "chi2_statistic": float, "chi2_p_value": float, "histogram_correlation": float, "lsb_embedding_suspected": bool },
    "G": { ... },
    "B": { ... }
  },

  "rs_analysis": {
    "R": { "R": float, "S": float, "R_minus": float, "S_minus": float, "estimated_payload_fraction": float },
    "G": { ... },
    "B": { ... }
  },

  "frequency_domain": {
    "R": { "noise_fft_total_energy": float, "low_freq_noise_ratio": float, "high_freq_noise_ratio": float, "noise_std": float },
    "G": { ... },
    "B": { ... }
  },

  "secret_vs_cover": {            // only if --secret provided
    "psnr_db": float,
    "ssim": float,
    "mse": float,
    "mean_diff": float
  },

  "verdict": {
    "confidence": "HIGH | MEDIUM | LOW",
    "stego_suspected": bool,
    "flags": ["...", "..."]
  },

  "analysis_time_sec": float
}
```

---

## Dependency Map

```
numpy          — all array operations, FFT fallback
Pillow         — image loading and format handling
scipy          — chi-square CDF, FFT (preferred over numpy FFT)
scikit-image   — PSNR and SSIM (preferred; falls back to manual impl)
matplotlib     — plot generation (optional, skipped if absent)
rich           — terminal report rendering (optional, falls back to JSON dump)
```

All dependencies degrade gracefully — the core metrics run with only numpy and Pillow.

---

## Mode-Specific Interpretation Guide

### LSB output (Sem 6)
The strongest signals are Chi-square (should fire on multiple channels for full payloads) and RS analysis (estimated payload should track the actual embedding rate closely). BER will be high (~LSB_bits / total_bits). Bit-plane 0 flip rate ≈ 0.5 for a full payload.

### CNN output (Sem 7)
Chi-square and RS are unlikely to fire (CNN does not produce clean ±1 changes). The relevant signals are SSIM (encoder quality), noise distribution shape (should be Gaussian-like), frequency domain noise ratio (smooth encoder → more low-freq noise). BER will be high in absolute value but spread across multiple bit planes.

### INR output (Sem 8)
Similar to CNN in that classical LSB detectors should not fire. PSNR and SSIM are the primary imperceptibility benchmarks. The frequency domain FFT pattern is the most distinctive — INR coordinate networks can produce spatially structured noise patterns unlike either LSB (white noise) or CNN (smooth noise). The secret vs cover comparison is most relevant here since INR is cross-modal.
