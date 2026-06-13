"""
╔══════════════════════════════════════════════════════════════════╗
║         UNIFIED STEGANALYSIS TOOLKIT — Group 02, IUST Kashmir    ║
║         Cross-Modal Steganography Final Year Project             ║
║         Covers: LSB (Sem 6) · CNN (Sem 7) · INR (Sem 8)          ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python steganalysis.py --cover <path> --stego <path> [options]

    --mode        lsb | cnn | inr | auto   (default: auto)
    --secret      path to secret/payload image (optional, for extra metrics)
    --output      path to save JSON report   (optional)
    --plot        save comparison plots      (optional, pass filename prefix)
    --quiet       suppress rich console output

Examples:
    python steganalysis.py --cover cover.png --stego stego.png
    python steganalysis.py --cover c.png --stego s.png --secret sec.png --plot results
    python steganalysis.py --cover c.png --stego s.png --mode inr --output report.json
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ── Optional imports (graceful degradation) ─────────────────────────────────
try:
    from scipy.stats import chi2, entropy as scipy_entropy
    from scipy.fft import fft2, fftshift
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_OK = True
except ImportError:
    RICH_OK = False


# ════════════════════════════════════════════════════════════════════════════
#  CORE METRIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """Load image as float32 numpy array [0,255]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def mse(cover: np.ndarray, stego: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((cover - stego) ** 2))


def psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher = better imperceptibility."""
    err = mse(cover, stego)
    if err == 0.0:
        return float("inf")
    if SKIMAGE_OK:
        return float(psnr_fn(cover, stego, data_range=255.0))
    return 10 * math.log10((255.0 ** 2) / err)


def ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM). Range [-1, 1]; 1 = identical.
    Captures luminance, contrast, and structure simultaneously.
    """
    if SKIMAGE_OK:
        return float(ssim_fn(
            cover.astype(np.uint8),
            stego.astype(np.uint8),
            channel_axis=2,
            data_range=255
        ))
    # Fallback: simplified per-channel mean SSIM
    scores = []
    for c in range(cover.shape[2]):
        mu1, mu2 = cover[:, :, c].mean(), stego[:, :, c].mean()
        s1 = cover[:, :, c].std(); s2 = stego[:, :, c].std()
        cov = np.mean((cover[:, :, c] - mu1) * (stego[:, :, c] - mu2))
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
        den = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
        scores.append(num / den if den != 0 else 0.0)
    return float(np.mean(scores))


def uqi(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Universal Quality Index (UQI / Q-index).
    Decomposes distortion into: loss of correlation, luminance distortion,
    contrast distortion. Range [-1, 1]; 1 = perfect.
    """
    scores = []
    for c in range(cover.shape[2]):
        x = cover[:, :, c].flatten().astype(np.float64)
        y = stego[:, :, c].flatten().astype(np.float64)
        mx, my = x.mean(), y.mean()
        sx = x.std(ddof=1); sy = y.std(ddof=1)
        sxy = np.mean((x - mx) * (y - my))
        q = (4 * sxy * mx * my) / ((sx**2 + sy**2) * (mx**2 + my**2 + 1e-10) + 1e-10)
        scores.append(q)
    return float(np.mean(scores))


def ncc(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Normalized Cross-Correlation.
    Used in watermarking research; robust to brightness shifts.
    Range [0, 2]; 1.0 = identical; >1 or <1 = deviation.
    """
    scores = []
    for c in range(cover.shape[2]):
        x = cover[:, :, c].flatten()
        y = stego[:, :, c].flatten()
        denom = (np.sum(x ** 2) + 1e-10)
        scores.append(float(np.sum(x * y) / denom))
    return float(np.mean(scores))


def snr(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio (dB).
    Unlike PSNR, uses actual signal power (not max^2) as reference.
    """
    noise = cover - stego
    signal_power = np.mean(cover ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * math.log10(signal_power / noise_power))


def bit_error_rate(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Bit Error Rate — fraction of differing bits in the 8-bit pixel values.
    Relevant for LSB analysis.
    """
    c = cover.astype(np.uint8)
    s = stego.astype(np.uint8)
    total_bits = c.size * 8
    xor = np.bitwise_xor(c, s)
    differing_bits = sum(bin(b).count('1') for b in xor.flatten())
    return float(differing_bits / total_bits)


def payload_capacity(cover: np.ndarray, mode: str) -> dict:
    """
    Theoretical max embedding capacity for different methods.
    Returns bits, bytes, and bpp (bits per pixel).
    """
    h, w, ch = cover.shape
    total_pixels = h * w
    caps = {}
    if mode in ("lsb", "auto"):
        for n_lsb in [1, 2, 3, 4]:
            bits = total_pixels * ch * n_lsb
            caps[f"lsb_{n_lsb}bit"] = {
                "bits": bits, "bytes": bits // 8, "bpp": n_lsb * ch
            }
    if mode in ("cnn", "inr", "auto"):
        # Deep methods typically achieve ~1 bpp on one channel
        bits = total_pixels  # conservative
        caps["deep_1bpp"] = {"bits": bits, "bytes": bits // 8, "bpp": 1}
        bits2 = total_pixels * ch
        caps["deep_3bpp"] = {"bits": bits2, "bytes": bits2 // 8, "bpp": 3}
    return caps


# ── Histogram & Statistical Analysis ────────────────────────────────────────

def histogram_analysis(cover: np.ndarray, stego: np.ndarray) -> dict:
    """Chi-square test on LSB histogram pairs (classical LSB detection)."""
    results = {}
    channel_names = ["R", "G", "B"]
    for i, ch_name in enumerate(channel_names):
        c = cover[:, :, i].astype(np.uint8).flatten()
        s = stego[:, :, i].astype(np.uint8).flatten()

        # Histogram
        cover_hist, _ = np.histogram(c, bins=256, range=(0, 255))
        stego_hist, _ = np.histogram(s, bins=256, range=(0, 255))

        # Chi-square: pair values (0,1), (2,3), ..., (254,255)
        # If LSB is randomly flipped, adjacent pairs should be equalised
        chi2_stat = 0.0
        pairs = 0
        for v in range(0, 256, 2):
            observed_0 = stego_hist[v]
            observed_1 = stego_hist[v + 1] if v + 1 < 256 else 0
            expected = (observed_0 + observed_1) / 2.0
            if expected > 0:
                chi2_stat += (observed_0 - expected) ** 2 / expected
                chi2_stat += (observed_1 - expected) ** 2 / expected
                pairs += 1

        # p-value (low p → likely stego)
        if SCIPY_OK and pairs > 0:
            p_val = float(1 - chi2.cdf(chi2_stat, df=pairs))
        else:
            p_val = None

        # Histogram correlation
        if cover_hist.sum() > 0 and stego_hist.sum() > 0:
            corr = float(np.corrcoef(cover_hist, stego_hist)[0, 1])
        else:
            corr = None

        results[ch_name] = {
            "chi2_statistic": round(chi2_stat, 4),
            "chi2_p_value": round(p_val, 6) if p_val is not None else "N/A",
            "histogram_correlation": round(corr, 6) if corr is not None else "N/A",
            "lsb_embedding_suspected": (p_val < 0.05) if p_val is not None else False,
        }
    return results


def entropy_analysis(cover: np.ndarray, stego: np.ndarray) -> dict:
    """Shannon entropy per channel. Stego images often have higher entropy."""
    results = {}
    channel_names = ["R", "G", "B"]
    for i, ch_name in enumerate(channel_names):
        def chan_entropy(arr):
            flat = arr[:, :, i].astype(np.uint8).flatten()
            hist, _ = np.histogram(flat, bins=256, range=(0, 255), density=True)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log2(hist)))

        cover_ent = chan_entropy(cover)
        stego_ent = chan_entropy(stego)
        results[ch_name] = {
            "cover_entropy_bits": round(cover_ent, 4),
            "stego_entropy_bits": round(stego_ent, 4),
            "entropy_delta": round(stego_ent - cover_ent, 6),
        }
    return results


def lsb_plane_analysis(cover: np.ndarray, stego: np.ndarray) -> dict:
    """
    Analyse each bit-plane (0=LSB to 7=MSB).
    LSB steganography shows high randomness in bit-plane 0.
    """
    results = {}
    channel_names = ["R", "G", "B"]
    for i, ch_name in enumerate(channel_names):
        c = cover[:, :, i].astype(np.uint8)
        s = stego[:, :, i].astype(np.uint8)
        planes = {}
        for bit in range(8):
            cover_plane = (c >> bit) & 1
            stego_plane = (s >> bit) & 1
            # Randomness of stego plane (0.5 = perfectly random)
            stego_mean = float(stego_plane.mean())
            plane_diff_rate = float(np.mean(cover_plane != stego_plane))
            planes[f"bit_{bit}"] = {
                "cover_ones_ratio": round(float(cover_plane.mean()), 4),
                "stego_ones_ratio": round(stego_mean, 4),
                "flip_rate": round(plane_diff_rate, 4),
            }
        results[ch_name] = planes
    return results


def rs_analysis(img_arr: np.ndarray) -> dict:
    """
    RS (Regular-Singular) Steganalysis — Fridrich et al.
    Estimates LSB payload fraction. ~0.0 = clean; ~0.5 = fully embedded.
    """
    results = {}
    channel_names = ["R", "G", "B"]

    def flipping_function(group):
        """F1: flip LSB of every other pixel."""
        g = group.copy().astype(np.int32)
        g[1::2] ^= 1
        return g

    def discriminant(group):
        """Count sign changes (measure of smoothness)."""
        diff = np.diff(group.astype(np.int32))
        return float(np.sum(np.abs(diff)))

    for i, ch_name in enumerate(channel_names):
        ch = img_arr[:, :, i].astype(np.uint8)
        h, w = ch.shape
        R, S, R_m, S_m = 0, 0, 0, 0
        count = 0

        for row in range(h):
            for col in range(0, w - 3, 4):
                g = ch[row, col:col + 4]
                f_g = flipping_function(g)
                d_g = discriminant(g)
                d_fg = discriminant(f_g)

                # Regular/Singular for +F
                if d_fg > d_g: R += 1
                elif d_fg < d_g: S += 1

                # Flipped groups: apply -F (flip LSB of original first)
                g_m = g.copy(); g_m ^= 1
                f_gm = flipping_function(g_m)
                d_gm = discriminant(g_m)
                d_fgm = discriminant(f_gm)

                if d_fgm > d_gm: R_m += 1
                elif d_fgm < d_gm: S_m += 1
                count += 1

        if count == 0:
            results[ch_name] = {"estimated_payload_fraction": 0.0}
            continue

        R /= count; S /= count; R_m /= count; S_m /= count

        # Solve quadratic for p (payload fraction)
        a = 2 * (R_m - R)
        b = R - R_m + S_m - S
        if abs(a) < 1e-10:
            p = 0.0
        else:
            disc = b ** 2 - 2 * a * (S_m - S)
            p = (-b - math.sqrt(max(0, disc))) / a if disc >= 0 else 0.0
            p = min(max(p, 0.0), 1.0)

        results[ch_name] = {
            "R": round(R, 4), "S": round(S, 4),
            "R_minus": round(R_m, 4), "S_minus": round(S_m, 4),
            "estimated_payload_fraction": round(p, 4),
        }
    return results


def frequency_domain_analysis(cover: np.ndarray, stego: np.ndarray) -> dict:
    """
    DCT/FFT-based analysis.
    CNN and deep stego can leave artifacts in frequency domain.
    """
    results = {}
    noise = cover - stego
    channel_names = ["R", "G", "B"]

    for i, ch_name in enumerate(channel_names):
        n = noise[:, :, i]
        if SCIPY_OK:
            N_fft = np.abs(fftshift(fft2(n)))
        else:
            N_fft = np.abs(np.fft.fftshift(np.fft.fft2(n)))

        h, w = N_fft.shape
        cy, cx = h // 2, w // 2

        # Energy in concentric frequency bands
        total_energy = float(np.sum(N_fft ** 2))
        center_r = min(cy, cx) // 4

        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

        low_mask = dist <= center_r
        low_energy = float(np.sum((N_fft * low_mask) ** 2))
        high_energy = total_energy - low_energy

        results[ch_name] = {
            "noise_fft_total_energy": round(total_energy, 2),
            "low_freq_noise_ratio": round(low_energy / (total_energy + 1e-10), 4),
            "high_freq_noise_ratio": round(high_energy / (total_energy + 1e-10), 4),
            "noise_max": round(float(n.max()), 4),
            "noise_min": round(float(n.min()), 4),
            "noise_std": round(float(n.std()), 6),
        }
    return results


def pixel_difference_stats(cover: np.ndarray, stego: np.ndarray) -> dict:
    """Detailed pixel-level difference statistics."""
    diff = np.abs(cover - stego)
    noise = cover - stego
    return {
        "max_absolute_diff": float(diff.max()),
        "mean_absolute_diff": round(float(diff.mean()), 6),
        "median_absolute_diff": round(float(np.median(diff)), 4),
        "std_diff": round(float(diff.std()), 6),
        "changed_pixels_count": int(np.sum(diff > 0)),
        "changed_pixels_pct": round(float(np.mean(diff > 0)) * 100, 4),
        "pixels_diff_by_1": int(np.sum(diff == 1)),
        "pixels_diff_gt_1": int(np.sum(diff > 1)),
        "noise_skewness": round(float(
            np.mean(((noise - noise.mean()) / (noise.std() + 1e-10)) ** 3)
        ), 6),
        "noise_kurtosis": round(float(
            np.mean(((noise - noise.mean()) / (noise.std() + 1e-10)) ** 4)
        ), 6),
    }


def steganalysis_verdict(metrics: dict, mode: str) -> dict:
    """
    Rule-based steganalysis verdict.
    Uses computed metrics to flag likely steganographic content.
    """
    flags = []
    confidence = "LOW"

    psnr_val = metrics["imperceptibility"]["psnr_db"]
    ssim_val = metrics["imperceptibility"]["ssim"]
    ber = metrics["imperceptibility"]["bit_error_rate"]

    # General thresholds (academic literature)
    if isinstance(psnr_val, float):
        if psnr_val < 30:
            flags.append(f"LOW PSNR ({psnr_val:.1f} dB) — perceptible distortion")
        elif psnr_val < 40:
            flags.append(f"MODERATE PSNR ({psnr_val:.1f} dB) — moderate embedding")
        else:
            flags.append(f"HIGH PSNR ({psnr_val:.1f} dB) — good imperceptibility")

    if ssim_val < 0.95:
        flags.append(f"SSIM {ssim_val:.4f} < 0.95 — structural changes detected")
    if ber > 0.001:
        flags.append(f"BER {ber:.4f} — significant bit-level changes")

    # Mode-specific
    if mode in ("lsb", "auto"):
        chi_flags = sum(
            1 for ch in metrics.get("histogram_chi2", {}).values()
            if ch.get("lsb_embedding_suspected", False)
        )
        if chi_flags >= 2:
            flags.append(f"Chi-square test: LSB embedding detected in {chi_flags}/3 channels")
            confidence = "HIGH"

        rs = metrics.get("rs_analysis", {})
        avg_payload = np.mean([
            v.get("estimated_payload_fraction", 0)
            for v in rs.values() if isinstance(v, dict)
        ])
        if avg_payload > 0.05:
            flags.append(f"RS Analysis: ~{avg_payload*100:.1f}% payload embedded")
            confidence = "HIGH"

    # Frequency domain
    freq = metrics.get("frequency_domain", {})
    if freq:
        high_ratios = [
            v.get("high_freq_noise_ratio", 0)
            for v in freq.values() if isinstance(v, dict)
        ]
        if any(r > 0.7 for r in high_ratios):
            flags.append("High-frequency noise dominance — possible deep stego artifact")

    if len(flags) >= 3 and confidence == "LOW":
        confidence = "MEDIUM"

    return {
        "confidence": confidence,
        "flags": flags,
        "stego_suspected": len(flags) > 1,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def generate_plots(cover: np.ndarray, stego: np.ndarray,
                   metrics: dict, prefix: str, secret: np.ndarray = None):
    if not MATPLOTLIB_OK:
        print("[WARN] matplotlib not available — skipping plots")
        return

    cover_u = cover.astype(np.uint8)
    stego_u = stego.astype(np.uint8)
    diff = np.abs(cover - stego).astype(np.uint8)
    diff_amp = np.clip(diff * 20, 0, 255).astype(np.uint8)  # amplified

    n_cols = 5 if secret is None else 6
    fig = plt.figure(figsize=(n_cols * 3.5, 12))
    gs = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.45, wspace=0.3)
    fig.patch.set_facecolor("#0f0f1a")

    def ax_style(ax, title):
        ax.set_title(title, color="#e0e0ff", fontsize=8, pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Row 0: images
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(cover_u); ax_style(ax, "Cover Image")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(stego_u); ax_style(ax, "Stego Image")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(diff_amp); ax_style(ax, "Diff × 20 (Amplified)")

    # LSB planes
    ax = fig.add_subplot(gs[0, 3])
    lsb_cover = (cover_u[:, :, 1] & 1) * 255
    ax.imshow(lsb_cover, cmap="gray"); ax_style(ax, "Cover LSB Plane (G)")

    ax = fig.add_subplot(gs[0, 4])
    lsb_stego = (stego_u[:, :, 1] & 1) * 255
    ax.imshow(lsb_stego, cmap="gray"); ax_style(ax, "Stego LSB Plane (G)")

    if secret is not None:
        ax = fig.add_subplot(gs[0, 5])
        ax.imshow(secret.astype(np.uint8)); ax_style(ax, "Secret Image")

    # Row 1: histograms
    channel_colors = ["#ff4444", "#44ff44", "#4488ff"]
    channel_names = ["R", "G", "B"]
    for ci, (cname, col) in enumerate(zip(channel_names, channel_colors)):
        ax = fig.add_subplot(gs[1, ci])
        ax.set_facecolor("#111122")
        ch_c = cover_u[:, :, ci].flatten()
        ch_s = stego_u[:, :, ci].flatten()
        hist_c, bins = np.histogram(ch_c, bins=256, range=(0, 255))
        hist_s, _ = np.histogram(ch_s, bins=256, range=(0, 255))
        ax.plot(bins[:-1], hist_c, color=col, alpha=0.7, linewidth=0.8, label="cover")
        ax.plot(bins[:-1], hist_s, color="white", alpha=0.5, linewidth=0.8, label="stego")
        ax.set_title(f"Histogram — {cname}", color="#e0e0ff", fontsize=8, pad=4)
        ax.tick_params(colors="#888", labelsize=6)
        ax.legend(fontsize=5, facecolor="#1a1a2e", labelcolor="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # Noise distribution
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor("#111122")
    noise_flat = (cover - stego).flatten()
    ax.hist(noise_flat, bins=100, color="#aa66ff", alpha=0.8, edgecolor="none")
    ax.set_title("Noise Distribution", color="#e0e0ff", fontsize=8, pad=4)
    ax.axvline(0, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.tick_params(colors="#888", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # Bit-plane flip rates
    ax = fig.add_subplot(gs[1, 4])
    ax.set_facecolor("#111122")
    lsb_data = metrics.get("lsb_planes", {})
    if lsb_data:
        for ci, (ch, col) in enumerate(zip(channel_names, channel_colors)):
            if ch in lsb_data:
                flip_rates = [lsb_data[ch].get(f"bit_{b}", {}).get("flip_rate", 0) for b in range(8)]
                ax.plot(range(8), flip_rates, marker='o', markersize=3,
                        color=col, linewidth=1, label=ch, alpha=0.85)
    ax.set_title("Bit-Plane Flip Rates", color="#e0e0ff", fontsize=8, pad=4)
    ax.set_xlabel("Bit Plane (0=LSB)", color="#888", fontsize=6)
    ax.tick_params(colors="#888", labelsize=6)
    ax.legend(fontsize=5, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    if secret is not None:
        ax = fig.add_subplot(gs[1, 5])
        ax.set_facecolor("#111122")
        diff_sec = np.abs(cover.astype(np.float32) - secret.astype(np.float32)).flatten()
        ax.hist(diff_sec, bins=80, color="#ff8844", alpha=0.8, edgecolor="none")
        ax.set_title("Cover vs Secret Diff", color="#e0e0ff", fontsize=8, pad=4)
        ax.tick_params(colors="#888", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # Row 2: FFT noise spectrum + metric bar chart + RS chart
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor("#111122")
    noise_g = (cover[:, :, 1] - stego[:, :, 1])
    fft_noise = np.abs(np.fft.fftshift(np.fft.fft2(noise_g)))
    fft_log = np.log1p(fft_noise)
    ax.imshow(fft_log, cmap="magma"); ax_style(ax, "FFT of Noise (G channel)")

    # Metric summary bar
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor("#111122")
    imp = metrics["imperceptibility"]
    metric_keys = ["psnr_db", "ssim", "uqi", "ncc", "snr_db"]
    metric_labels = ["PSNR/50", "SSIM", "UQI", "NCC", "SNR/50"]
    normalized = []
    for k, lbl in zip(metric_keys, metric_labels):
        v = imp.get(k, 0)
        if isinstance(v, str): v = 0
        if "psnr" in k or "snr" in k: v = min(v, 50) / 50
        normalized.append(max(0, min(1, float(v))))

    bars = ax.barh(metric_labels, normalized,
                   color=["#4af", "#f4a", "#4fa", "#fa4", "#a4f"], alpha=0.8)
    ax.set_xlim(0, 1.1)
    ax.axvline(1.0, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_title("Imperceptibility (Normalised)", color="#e0e0ff", fontsize=8, pad=4)
    ax.tick_params(colors="#888", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # RS payload estimates
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor("#111122")
    rs_data = metrics.get("rs_analysis", {})
    if rs_data:
        chs = [c for c in rs_data.keys() if isinstance(rs_data[c], dict)]
        payloads = [rs_data[c].get("estimated_payload_fraction", 0) * 100 for c in chs]
        ax.bar(chs, payloads, color=channel_colors[:len(chs)], alpha=0.8)
        ax.axhline(5, color="yellow", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_ylabel("Payload %", color="#888", fontsize=6)
    ax.set_title("RS Analysis — Payload Est.", color="#e0e0ff", fontsize=8, pad=4)
    ax.tick_params(colors="#888", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # Entropy delta
    ax = fig.add_subplot(gs[2, 3])
    ax.set_facecolor("#111122")
    ent_data = metrics.get("entropy", {})
    if ent_data:
        chs = list(ent_data.keys())
        cover_ents = [ent_data[c]["cover_entropy_bits"] for c in chs]
        stego_ents = [ent_data[c]["stego_entropy_bits"] for c in chs]
        x = np.arange(len(chs))
        ax.bar(x - 0.2, cover_ents, 0.35, label="cover", color="#4af", alpha=0.8)
        ax.bar(x + 0.2, stego_ents, 0.35, label="stego", color="#f4a", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(chs, color="#888", fontsize=6)
        ax.legend(fontsize=5, facecolor="#1a1a2e", labelcolor="white")
    ax.set_title("Shannon Entropy (bits)", color="#e0e0ff", fontsize=8, pad=4)
    ax.tick_params(colors="#888", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # Verdict panel
    ax = fig.add_subplot(gs[2, 4])
    ax.set_facecolor("#111122")
    verdict = metrics.get("verdict", {})
    ax.axis("off")
    conf = verdict.get("confidence", "N/A")
    color_map = {"HIGH": "#ff4444", "MEDIUM": "#ffaa44", "LOW": "#44ff88"}
    vc = color_map.get(conf, "white")
    ax.text(0.5, 0.85, f"Confidence: {conf}", ha="center", va="top",
            transform=ax.transAxes, color=vc, fontsize=10, fontweight="bold")
    ax.text(0.5, 0.7, f"Stego Suspected: {verdict.get('stego_suspected', False)}",
            ha="center", va="top", transform=ax.transAxes, color="white", fontsize=8)
    flags = verdict.get("flags", [])
    for j, flag in enumerate(flags[:4]):
        ax.text(0.05, 0.55 - j * 0.13, f"• {flag[:55]}",
                ha="left", va="top", transform=ax.transAxes,
                color="#ccccee", fontsize=5.5, wrap=True)
    ax.set_title("Steganalysis Verdict", color="#e0e0ff", fontsize=8, pad=4)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    if secret is not None and n_cols > 5:
        ax = fig.add_subplot(gs[2, 5])
        ax.set_facecolor("#111122")
        # Cover vs secret scatter (sample)
        n_sample = min(2000, cover[:, :, 1].size)
        idx = np.random.choice(cover[:, :, 1].size, n_sample, replace=False)
        ax.scatter(cover[:, :, 1].flatten()[idx],
                   secret[:, :, 1].flatten()[idx],
                   alpha=0.15, s=1, color="#88ffcc")
        ax.set_xlabel("Cover G", color="#888", fontsize=6)
        ax.set_ylabel("Secret G", color="#888", fontsize=6)
        ax.set_title("Cover vs Secret (G)", color="#e0e0ff", fontsize=8, pad=4)
        ax.tick_params(colors="#888", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    psnr_val = metrics["imperceptibility"]["psnr_db"]
    psnr_str = f"{psnr_val:.2f}" if isinstance(psnr_val, float) else "∞"
    ssim_str = f"{metrics['imperceptibility']['ssim']:.4f}"
    fig.suptitle(
        f"Steganalysis Report — Mode: {metrics['mode'].upper()} | "
        f"PSNR: {psnr_str} dB | SSIM: {ssim_str}",
        color="white", fontsize=11, y=0.98, fontweight="bold"
    )

    out_path = f"{prefix}_steganalysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")
    plt.close()
    print(f"[PLOT] Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def detect_mode(cover: np.ndarray, stego: np.ndarray) -> str:
    """
    Heuristic mode detection:
    - LSB: lots of pixels changed by exactly 1, mostly in low bit-planes
    - CNN/INR: smooth noise, potential high-freq artifacts, larger per-pixel diffs
    """
    diff = np.abs(cover - stego)
    changed = np.sum(diff > 0)
    if changed == 0:
        return "lsb"  # identical images
    changed_by_1 = np.sum(diff == 1)
    ratio = changed_by_1 / max(changed, 1)

    c_u = cover.astype(np.uint8)
    s_u = stego.astype(np.uint8)
    lsb_changes = np.sum((c_u ^ s_u) & 1)
    total_pixels = c_u.size

    if ratio > 0.9 and lsb_changes / total_pixels > 0.1:
        return "lsb"
    elif diff.std() < 2.0:
        return "cnn"
    else:
        return "inr"


def run_analysis(cover_path: str, stego_path: str,
                 secret_path: str = None, mode: str = "auto",
                 plot_prefix: str = None, quiet: bool = False) -> dict:

    t0 = time.time()
    cover = load_image(cover_path)
    stego = load_image(stego_path)

    # Resize stego to cover if needed
    if cover.shape != stego.shape:
        stego_pil = Image.open(stego_path).convert("RGB").resize(
            (cover.shape[1], cover.shape[0]), Image.LANCZOS
        )
        stego = np.array(stego_pil, dtype=np.float32)

    secret = load_image(secret_path) if secret_path else None
    if secret is not None and secret.shape != cover.shape:
        sec_pil = Image.open(secret_path).convert("RGB").resize(
            (cover.shape[1], cover.shape[0]), Image.LANCZOS
        )
        secret = np.array(sec_pil, dtype=np.float32)

    detected_mode = detect_mode(cover, stego) if mode == "auto" else mode

    # ── Build metrics dict ────────────────────────────────────────────────
    metrics = {
        "mode": detected_mode,
        "cover_path": cover_path,
        "stego_path": stego_path,
        "secret_path": secret_path,
        "image_shape": list(cover.shape),
        "image_size_px": cover.shape[0] * cover.shape[1],
    }

    # Imperceptibility
    psnr_v = psnr(cover, stego)
    metrics["imperceptibility"] = {
        "psnr_db": round(psnr_v, 4) if not math.isinf(psnr_v) else "inf",
        "ssim": round(ssim(cover, stego), 6),
        "mse": round(mse(cover, stego), 6),
        "snr_db": round(snr(cover, stego), 4),
        "uqi": round(uqi(cover, stego), 6),
        "ncc": round(ncc(cover, stego), 6),
        "bit_error_rate": round(bit_error_rate(cover, stego), 8),
    }

    # Capacity
    metrics["capacity"] = payload_capacity(cover, detected_mode)

    # Pixel-level diff
    metrics["pixel_differences"] = pixel_difference_stats(cover, stego)

    # Entropy
    metrics["entropy"] = entropy_analysis(cover, stego)

    # LSB planes
    metrics["lsb_planes"] = lsb_plane_analysis(cover, stego)

    # Chi-square histogram test
    metrics["histogram_chi2"] = histogram_analysis(cover, stego)

    # RS analysis (can be slow on large images — sample if needed)
    h, w = cover.shape[:2]
    if h * w > 512 * 512:
        # Sample a 512×512 region from center for RS
        cy, cx = h // 2, w // 2
        crop_c = cover[cy-256:cy+256, cx-256:cx+256]
        crop_s = stego[cy-256:cy+256, cx-256:cx+256]
        metrics["rs_analysis"] = rs_analysis(crop_c)
        metrics["rs_analysis"]["_note"] = "Computed on 512×512 center crop for performance"
    else:
        metrics["rs_analysis"] = rs_analysis(cover)

    # Frequency domain
    metrics["frequency_domain"] = frequency_domain_analysis(cover, stego)

    # Secret vs cover (if provided)
    if secret is not None:
        metrics["secret_vs_cover"] = {
            "psnr_db": round(psnr(cover, secret), 4),
            "ssim": round(ssim(cover, secret), 6),
            "mse": round(mse(cover, secret), 6),
            "mean_diff": round(float(np.abs(cover - secret).mean()), 4),
        }

    # Verdict
    metrics["verdict"] = steganalysis_verdict(metrics, detected_mode)
    metrics["analysis_time_sec"] = round(time.time() - t0, 3)

    # Plot
    if plot_prefix:
        generate_plots(cover, stego, metrics, plot_prefix, secret)

    return metrics


# ════════════════════════════════════════════════════════════════════════════
#  RICH CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict):
    if not RICH_OK:
        print(json.dumps(metrics, indent=2))
        return

    console = Console()
    imp = metrics["imperceptibility"]
    verdict = metrics["verdict"]

    # Header
    psnr_v = imp['psnr_db']
    psnr_str = f"{psnr_v:.2f}" if isinstance(psnr_v, float) else "∞"

    conf_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(
        verdict["confidence"], "white"
    )

    console.print(Panel(
        f"[bold white]UNIFIED STEGANALYSIS REPORT[/bold white]\n"
        f"[dim]Group 02 · IUST Kashmir · Cross-Modal Steganography[/dim]\n"
        f"Mode: [cyan]{metrics['mode'].upper()}[/cyan]  |  "
        f"Image: [dim]{metrics['image_shape'][1]}×{metrics['image_shape'][0]}[/dim]  |  "
        f"Time: [dim]{metrics['analysis_time_sec']}s[/dim]",
        border_style="bright_blue"
    ))

    # ── Imperceptibility Table ──
    t = Table(title="Imperceptibility Metrics", box=box.ROUNDED,
              border_style="blue", header_style="bold cyan")
    t.add_column("Metric", style="white")
    t.add_column("Value", justify="right")
    t.add_column("Threshold", justify="center", style="dim")
    t.add_column("Status", justify="center")

    def status(val, good, ok, higher_is_better=True):
        if isinstance(val, str): return "[white]N/A[/white]"
        if higher_is_better:
            if val >= good: return "[green]✓ GOOD[/green]"
            if val >= ok:   return "[yellow]~ OK[/yellow]"
            return "[red]✗ POOR[/red]"
        else:
            if val <= good: return "[green]✓ GOOD[/green]"
            if val <= ok:   return "[yellow]~ OK[/yellow]"
            return "[red]✗ POOR[/red]"

    psnr_disp = f"{psnr_v:.2f} dB" if isinstance(psnr_v, float) else "∞ dB"
    t.add_row("PSNR",  psnr_disp,   ">40 / >30 dB", status(psnr_v if isinstance(psnr_v,float) else 999, 40, 30))
    t.add_row("SSIM",  f"{imp['ssim']:.6f}", ">0.99 / >0.95", status(imp['ssim'], 0.99, 0.95))
    t.add_row("MSE",   f"{imp['mse']:.4f}",  "<1.0 / <10.0", status(imp['mse'], 1.0, 10.0, False))
    t.add_row("SNR",   f"{imp['snr_db']:.2f} dB", ">30 / >20 dB", status(imp['snr_db'], 30, 20))
    t.add_row("UQI",   f"{imp['uqi']:.6f}",  ">0.99 / >0.95", status(imp['uqi'], 0.99, 0.95))
    t.add_row("NCC",   f"{imp['ncc']:.6f}",  ">0.999",        status(imp['ncc'], 0.999, 0.99))
    t.add_row("BER",   f"{imp['bit_error_rate']:.6f}", "<0.001", status(imp['bit_error_rate'], 0.001, 0.01, False))
    console.print(t)

    # ── Pixel Diff Table ──
    pd_t = Table(title="Pixel Difference Statistics", box=box.SIMPLE,
                 border_style="dim blue", header_style="bold cyan")
    pd_t.add_column("Statistic"); pd_t.add_column("Value", justify="right")
    pdiff = metrics["pixel_differences"]
    pd_t.add_row("Max Absolute Diff", str(pdiff["max_absolute_diff"]))
    pd_t.add_row("Mean Absolute Diff", str(pdiff["mean_absolute_diff"]))
    pd_t.add_row("Changed Pixels", f"{pdiff['changed_pixels_count']} ({pdiff['changed_pixels_pct']}%)")
    pd_t.add_row("Pixels Changed by ±1", str(pdiff["pixels_diff_by_1"]))
    pd_t.add_row("Pixels Changed by >1", str(pdiff["pixels_diff_gt_1"]))
    pd_t.add_row("Noise Skewness", str(pdiff["noise_skewness"]))
    pd_t.add_row("Noise Kurtosis", str(pdiff["noise_kurtosis"]))
    console.print(pd_t)

    # ── Chi-square Table ──
    chi_t = Table(title="Chi-Square Histogram Analysis (LSB Detection)",
                  box=box.SIMPLE, border_style="dim blue", header_style="bold cyan")
    chi_t.add_column("Channel"); chi_t.add_column("χ² Statistic", justify="right")
    chi_t.add_column("p-value", justify="right"); chi_t.add_column("Suspected?", justify="center")
    for ch, vals in metrics["histogram_chi2"].items():
        susp = "[red]YES[/red]" if vals.get("lsb_embedding_suspected") else "[green]NO[/green]"
        chi_t.add_row(ch, str(vals["chi2_statistic"]), str(vals["chi2_p_value"]), susp)
    console.print(chi_t)

    # ── RS Analysis Table ──
    rs_t = Table(title="RS Steganalysis (Payload Estimation)",
                 box=box.SIMPLE, border_style="dim blue", header_style="bold cyan")
    rs_t.add_column("Channel"); rs_t.add_column("R", justify="right")
    rs_t.add_column("S", justify="right"); rs_t.add_column("R−", justify="right")
    rs_t.add_column("S−", justify="right"); rs_t.add_column("Est. Payload %", justify="right")
    for ch, vals in metrics["rs_analysis"].items():
        if isinstance(vals, dict) and "estimated_payload_fraction" in vals:
            p = vals["estimated_payload_fraction"] * 100
            color = "red" if p > 5 else "green"
            rs_t.add_row(ch,
                str(vals.get("R", "—")), str(vals.get("S", "—")),
                str(vals.get("R_minus", "—")), str(vals.get("S_minus", "—")),
                f"[{color}]{p:.2f}%[/{color}]"
            )
    console.print(rs_t)

    # ── Entropy Table ──
    ent_t = Table(title="Shannon Entropy Analysis",
                  box=box.SIMPLE, border_style="dim blue", header_style="bold cyan")
    ent_t.add_column("Channel"); ent_t.add_column("Cover (bits)", justify="right")
    ent_t.add_column("Stego (bits)", justify="right"); ent_t.add_column("Δ", justify="right")
    for ch, vals in metrics["entropy"].items():
        delta = vals["entropy_delta"]
        col = "yellow" if abs(delta) > 0.001 else "green"
        ent_t.add_row(ch, str(vals["cover_entropy_bits"]),
                      str(vals["stego_entropy_bits"]),
                      f"[{col}]{delta:+.6f}[/{col}]")
    console.print(ent_t)

    # ── Capacity Table ──
    cap_t = Table(title="Embedding Capacity (Theoretical)",
                  box=box.SIMPLE, border_style="dim blue", header_style="bold cyan")
    cap_t.add_column("Method"); cap_t.add_column("Bits", justify="right")
    cap_t.add_column("Bytes", justify="right"); cap_t.add_column("BPP", justify="right")
    for method, vals in metrics["capacity"].items():
        cap_t.add_row(method, f"{vals['bits']:,}", f"{vals['bytes']:,}", str(vals["bpp"]))
    console.print(cap_t)

    # ── Secret vs Cover ──
    if "secret_vs_cover" in metrics:
        sv = metrics["secret_vs_cover"]
        console.print(Panel(
            f"[bold]Secret vs Cover Comparison[/bold]\n"
            f"PSNR: [cyan]{sv['psnr_db']} dB[/cyan]  |  "
            f"SSIM: [cyan]{sv['ssim']}[/cyan]  |  "
            f"MSE: [cyan]{sv['mse']}[/cyan]  |  "
            f"Mean Diff: [cyan]{sv['mean_diff']}[/cyan]",
            border_style="magenta"
        ))

    # ── Verdict ──
    flags_text = "\n".join(f"  • {f}" for f in verdict["flags"])
    suspected = "[red]YES — Steganographic content likely[/red]" if verdict["stego_suspected"] \
                else "[green]NO — No strong stego signals[/green]"
    console.print(Panel(
        f"[bold]Stego Suspected:[/bold] {suspected}\n"
        f"[bold]Confidence:[/bold] [{conf_color}]{verdict['confidence']}[/{conf_color}]\n\n"
        f"[bold]Flags:[/bold]\n{flags_text}",
        title="[bold white]STEGANALYSIS VERDICT[/bold white]",
        border_style=conf_color
    ))


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Steganalysis Toolkit — IUST Kashmir Group 02"
    )
    parser.add_argument("--cover",   required=True, help="Path to cover image")
    parser.add_argument("--stego",   required=True, help="Path to stego image")
    parser.add_argument("--secret",  default=None,  help="Path to secret image (optional)")
    parser.add_argument("--mode",    default="auto",
                        choices=["lsb", "cnn", "inr", "auto"],
                        help="Steganography mode (default: auto-detect)")
    parser.add_argument("--output",  default=None,  help="Save JSON report to this path")
    parser.add_argument("--plot",    default=None,  help="Save plots (provide filename prefix)")
    parser.add_argument("--quiet",   action="store_true", help="Suppress console output")
    args = parser.parse_args()

    # Validate paths
    for label, path in [("cover", args.cover), ("stego", args.stego)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} image not found: {path}")
            sys.exit(1)
    if args.secret and not os.path.exists(args.secret):
        print(f"[ERROR] secret image not found: {args.secret}")
        sys.exit(1)

    metrics = run_analysis(
        cover_path=args.cover,
        stego_path=args.stego,
        secret_path=args.secret,
        mode=args.mode,
        plot_prefix=args.plot,
        quiet=args.quiet,
    )

    if not args.quiet:
        print_report(metrics)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[SAVED] Report → {args.output}")


if __name__ == "__main__":
    main()
