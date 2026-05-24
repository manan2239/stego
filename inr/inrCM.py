"""
Cross-Modal INR Steganography
==============================
Hides one media modality inside another using Implicit Neural Representations.
The same network weights reconstruct both modalities — secret only with the key.

Supported modalities: image, audio, video, text (TXT)

USAGE:
  # Hide audio inside image
  python cross_modal_inr.py --modal1 image.png --modal2 audio.wav --mode hide --key 42

  # Extract secret back
  python cross_modal_inr.py --modal2 audio.wav --mode extract --key 42 --weights output/siren_weights.pth

  # Hide image inside video (low quality for CPU)
  python cross_modal_inr.py --modal1 video.mp4 --modal2 image.png --mode hide --key 7 --quality low

  # Override quality
  python cross_modal_inr.py --modal1 image.png --modal2 audio.wav --mode hide --key 42 --quality high
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import math
import json
import time
import hashlib
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Optional imports (graceful fallback) ─────────────────────────────────────
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not found. Image support disabled.")

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("[WARN] librosa/soundfile not found. Audio support disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] opencv-python not found. Video support disabled.")

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ═════════════════════════════════════════════════════════════════════════════
#  QUALITY PRESETS
# ═════════════════════════════════════════════════════════════════════════════
QUALITY_PRESETS = {
    "fast":   {"steps": 1000,  "hidden_dim": 128, "num_layers": 4, "omega_0": 30.0, "lr": 2e-4},
    "low":    {"steps": 3000,  "hidden_dim": 256, "num_layers": 5, "omega_0": 30.0, "lr": 2e-4},
    "medium": {"steps": 6000,  "hidden_dim": 512, "num_layers": 6, "omega_0": 30.0, "lr": 1e-4},
    "high":   {"steps": 12000, "hidden_dim": 512, "num_layers": 6, "omega_0": 30.0, "lr": 5e-5},
    "ultra":  {"steps": 20000, "hidden_dim": 1024,"num_layers": 7, "omega_0": 30.0, "lr": 2e-5},
}

def auto_quality(device):
    """Pick quality preset based on available hardware."""
    if device.type == "cuda":
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        if mem >= 8:   return "high"
        if mem >= 4:   return "medium"
        return "medium"
    return "low"  # CPU


# ═════════════════════════════════════════════════════════════════════════════
#  SIREN ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear  = nn.Linear(in_features, out_features)
        self._init_weights(is_first)

    def _init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class CrossModalSIREN(nn.Module):
    """
    Shared SIREN backbone with two output heads.
    Head 1 (cover)  → queried with original coordinates
    Head 2 (secret) → queried with key-shifted coordinates
    Both heads share all hidden weights — the secret is entangled in the weights.
    """
    def __init__(self, in_dim, out_dim_1, out_dim_2,
                 hidden_dim=256, num_layers=5, omega_0=30.0):
        super().__init__()

        # Shared backbone
        layers = [SirenLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.backbone = nn.Sequential(*layers)

        # Two output heads  (activation applied in forward methods)
        self.head_cover  = nn.Linear(hidden_dim, out_dim_1)
        self.head_secret = nn.Linear(hidden_dim, out_dim_2)

    def forward_cover(self, coords):
        return torch.sigmoid(self.head_cover(self.backbone(coords)))

    def forward_secret(self, coords):
        return torch.sigmoid(self.head_secret(self.backbone(coords)))


# ═════════════════════════════════════════════════════════════════════════════
#  KEY SYSTEM  — shifts coordinate space for secret queries
# ═════════════════════════════════════════════════════════════════════════════
def key_to_shift(key: int, coord_dim: int, device) -> torch.Tensor:
    """
    Converts integer key → deterministic coordinate shift vector.
    Without the correct key, secret extraction returns noise.
    """
    rng = np.random.RandomState(key)
    shift = rng.uniform(-0.5, 0.5, coord_dim).astype(np.float32)
    return torch.from_numpy(shift).to(device)

def apply_key(coords: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Shifts coordinates by key vector (wraps via tanh to stay in [-1,1])."""
    shifted = coords + shift.unsqueeze(0)
    return torch.tanh(shifted)


# ═════════════════════════════════════════════════════════════════════════════
#  MODALITY LOADERS  — returns (data_tensor, metadata_dict, coord_tensor)
# ═════════════════════════════════════════════════════════════════════════════
def detect_modality(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]:
        return "image"
    if ext in [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
        return "audio"
    if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return "video"
    if ext in [".txt"]:
        return "text"
    raise ValueError(f"Unsupported file type: {ext}")


# ── Image ─────────────────────────────────────────────────────────────────────
def load_image_modality(path, max_size=128):
    assert HAS_PIL, "Pillow required for image support. pip install Pillow"
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    W, H = img.size
    arr  = np.array(img, dtype=np.float32) / 255.0
    data = torch.from_numpy(arr).reshape(-1, 3)  # (H*W, 3)

    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)

    meta = {"type": "image", "H": H, "W": W, "path": path}
    return data, coords, meta


def save_image_modality(tensor, meta, out_path):
    H, W = meta["H"], meta["W"]
    arr  = (tensor.reshape(H, W, 3).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
    if HAS_PIL:
        Image.fromarray(arr).save(out_path)
    else:
        np.save(out_path.replace(".png", ".npy"), arr)


# ── Audio ─────────────────────────────────────────────────────────────────────
def load_audio_modality(path, max_samples=22050):
    assert HAS_AUDIO, "librosa + soundfile required. pip install librosa soundfile"
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y[:max_samples]                              # Truncate
    orig_min = float(y.min())                        # Save BEFORE normalizing
    orig_max = float(y.max())
    y = (y - orig_min) / (orig_max - orig_min + 1e-8)  # Normalize to [0,1]
    N    = len(y)
    data = torch.from_numpy(y).unsqueeze(-1)         # (N, 1)
    coords = torch.linspace(-1, 1, N).unsqueeze(-1)  # (N, 1)
    meta = {"type": "audio", "N": N, "sr": sr, "path": path,
            "orig_min": orig_min, "orig_max": orig_max}
    return data, coords, meta


def save_audio_modality(tensor, meta, out_path):
    assert HAS_AUDIO, "soundfile required for audio export."
    arr = tensor.reshape(-1).clamp(0, 1).cpu().numpy()
    # Denormalize
    arr = arr * (meta["orig_max"] - meta["orig_min"]) + meta["orig_min"]
    sf.write(out_path, arr, meta["sr"])


# ── Video ─────────────────────────────────────────────────────────────────────
def load_video_modality(path, max_frames=30, max_size=64):
    assert HAS_CV2, "opencv-python required for video support. pip install opencv-python"
    cap    = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (max_size, max_size))
        frames.append(frame)
    cap.release()

    T, H, W = len(frames), max_size, max_size
    arr  = np.array(frames, dtype=np.float32) / 255.0   # (T, H, W, 3)
    data = torch.from_numpy(arr).reshape(-1, 3)          # (T*H*W, 3)

    ts = torch.linspace(-1, 1, T)
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    gt, gy, gx = torch.meshgrid(ts, ys, xs, indexing='ij')
    coords = torch.stack([gx, gy, gt], dim=-1).reshape(-1, 3)

    meta = {"type": "video", "T": T, "H": H, "W": W, "path": path}
    return data, coords, meta


def save_video_modality(tensor, meta, out_path):
    assert HAS_CV2, "opencv required for video export."
    T, H, W = meta["T"], meta["H"], meta["W"]
    arr  = (tensor.reshape(T, H, W, 3).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(out_path, fourcc, 10, (W, H))
    for frame in arr:
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()


# ── Text ──────────────────────────────────────────────────────────────────────
# Encoding scheme: map the full printable range [32..126] + newline(10) + tab(9)
# to [0, 1].  We store the original ordinals so decode is exact (no clamp loss).
_TEXT_MIN_ORD = 9    # tab
_TEXT_MAX_ORD = 126  # '~'
_TEXT_RANGE   = _TEXT_MAX_ORD - _TEXT_MIN_ORD  # 117

def _ord_to_float(c: str) -> float:
    o = ord(c)
    # Clamp unknown chars to space (32) — keeps range valid
    o = max(_TEXT_MIN_ORD, min(_TEXT_MAX_ORD, o))
    return (o - _TEXT_MIN_ORD) / _TEXT_RANGE   # → [0, 1]

def _float_to_chr(v: float) -> str:
    o = int(round(v * _TEXT_RANGE)) + _TEXT_MIN_ORD
    o = max(_TEXT_MIN_ORD, min(_TEXT_MAX_ORD, o))
    return chr(o)

def load_text_modality(path, max_chars=4096):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()[:max_chars]
    chars  = [_ord_to_float(c) for c in text]
    N      = len(chars)
    data   = torch.tensor(chars, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    coords = torch.linspace(-1, 1, N).unsqueeze(-1)                  # (N, 1)
    meta   = {"type": "text", "N": N, "path": path, "text": text}
    return data, coords, meta


def save_text_modality(tensor, meta, out_path):
    arr   = tensor.reshape(-1).clamp(0, 1).cpu().numpy()
    chars = [_float_to_chr(float(v)) for v in arr]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(chars))


# ── Unified loader ────────────────────────────────────────────────────────────
def load_modality(path, device, max_size=128):
    mod = detect_modality(path)
    print(f"  Detected modality: {mod.upper()}  ({Path(path).name})")
    if   mod == "image": data, coords, meta = load_image_modality(path, max_size)
    elif mod == "audio": data, coords, meta = load_audio_modality(path)
    elif mod == "video": data, coords, meta = load_video_modality(path, max_size=max_size)
    elif mod == "text":  data, coords, meta = load_text_modality(path)
    return data.to(device), coords.to(device), meta


def save_modality(tensor, meta, out_path):
    mod = meta["type"]
    if   mod == "image": save_image_modality(tensor, meta, out_path)
    elif mod == "audio": save_audio_modality(tensor, meta, out_path)
    elif mod == "video": save_video_modality(tensor, meta, out_path)
    elif mod == "text":  save_text_modality(tensor, meta, out_path)
    print(f"  Saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return float('inf') if mse == 0 else 10 * math.log10(1.0 / mse)

def compute_ssim(pred, meta):
    """SSIM for images only."""
    if not HAS_SKIMAGE or meta["type"] != "image":
        return None
    H, W = meta["H"], meta["W"]
    p = pred.reshape(H, W, 3).clamp(0,1).cpu().numpy()
    return float(ssim_fn(p, p, channel_axis=2, data_range=1.0))  # placeholder


# ═════════════════════════════════════════════════════════════════════════════
#  HIDE MODE
# ═════════════════════════════════════════════════════════════════════════════
def hide(args, device, cfg):
    print(f"\n{'═'*58}")
    print(f"  MODE: HIDE  |  Key: {args.key}  |  Quality: {args.quality}")
    print(f"{'═'*58}")

    # Load both modalities
    print("\n  Loading modalities...")
    data1, coords1, meta1 = load_modality(args.modal1, device, args.max_size)
    data2, coords2, meta2 = load_modality(args.modal2, device, args.max_size)

    in_dim1  = coords1.shape[1]   # e.g. 2 for image, 1 for audio, 3 for video
    in_dim2  = coords2.shape[1]
    out_dim1 = data1.shape[1]     # e.g. 3 for RGB, 1 for audio
    out_dim2 = data2.shape[1]

    # Both modalities share the same coordinate dimensionality via padding
    in_dim = max(in_dim1, in_dim2)
    # Pad coords to same dim if needed
    def pad_coords(c, target_dim):
        if c.shape[1] < target_dim:
            pad = torch.zeros(c.shape[0], target_dim - c.shape[1], device=device)
            return torch.cat([c, pad], dim=1)
        return c
    coords1 = pad_coords(coords1, in_dim)
    coords2 = pad_coords(coords2, in_dim)

    # Key shift for secret coordinates
    shift = key_to_shift(args.key, in_dim, device)
    coords2_keyed = apply_key(coords2, shift)

    # Build model
    model = CrossModalSIREN(
        in_dim     = in_dim,
        out_dim_1  = out_dim1,
        out_dim_2  = out_dim2,
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        omega_0    = cfg["omega_0"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Modal 1 (cover)  : {meta1['type'].upper()}  — {data1.shape[0]:,} samples")
    print(f"  Modal 2 (secret) : {meta2['type'].upper()}  — {data2.shape[0]:,} samples")
    print(f"  Shared INR params: {total_params:,}")
    print(f"  Steps            : {cfg['steps']}")
    print(f"  Hidden dim       : {cfg['hidden_dim']}")
    print(f"  Device           : {device}")
    print(f"{'─'*58}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["steps"], eta_min=1e-6)
    loss_fn   = nn.MSELoss()

    # Adaptive loss weights based on modality types.
    # Text needs near-perfect fit (sharp char boundaries) → boost secret weight.
    # Audio also benefits from a higher secret weight for accurate waveform.
    # Cover (image/video) is perceptually forgiving at slightly lower weight.
    _MODALITY_WEIGHT = {"image": 1.0, "video": 1.0, "audio": 1.5, "text": 2.5}
    w_cover_base  = _MODALITY_WEIGHT.get(meta1["type"], 1.0)
    w_secret_base = _MODALITY_WEIGHT.get(meta2["type"], 1.0)
    total_w  = w_cover_base + w_secret_base
    w_cover  = w_cover_base  / total_w
    w_secret = w_secret_base / total_w
    print(f"  Loss weights — cover: {w_cover:.3f}  secret: {w_secret:.3f}  (adaptive)")

    print(f"  {'Step':>6}  {'Loss':>10}  {'PSNR Cover':>11}  {'PSNR Secret':>12}")
    print(f"  {'─'*45}")

    t_start = time.time()

    for step in range(1, cfg["steps"] + 1):
        model.train()
        optimizer.zero_grad()

        pred1 = model.forward_cover(coords1)
        pred2 = model.forward_secret(coords2_keyed)

        loss1 = loss_fn(pred1, data1)
        loss2 = loss_fn(pred2, data2)
        loss  = w_cover * loss1 + w_secret * loss2

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % (cfg["steps"] // 10) == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                p1 = compute_psnr(model.forward_cover(coords1), data1)
                p2 = compute_psnr(model.forward_secret(coords2_keyed), data2)
            print(f"  {step:>6}  {loss.item():>10.6f}  {p1:>11.2f}  {p2:>12.2f}")

    elapsed = time.time() - t_start

    # ── Final evaluation ─────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final1 = model.forward_cover(coords1)
        final2 = model.forward_secret(coords2_keyed)
        psnr1  = compute_psnr(final1, data1)
        psnr2  = compute_psnr(final2, data2)

    print(f"\n  ✓ Training complete in {elapsed:.1f}s")
    print(f"  PSNR Cover  : {psnr1:.2f} dB")
    print(f"  PSNR Secret : {psnr2:.2f} dB")
    # Warn immediately if secret PSNR is too low to be useful
    _SECRET_MIN = {"text": 30, "audio": 25}.get(meta2["type"], 28)
    if psnr2 < _SECRET_MIN:
        print(f"  ⚠  Secret PSNR too low ({psnr2:.1f} dB < {_SECRET_MIN} dB threshold).")
        if args.quality == "fast":
            print(f"     'fast' preset is intentionally minimal — use --quality low or higher for usable output.")
        else:
            print(f"     Output will be corrupted. Re-run with --quality medium or higher.")
    else:
        print(f"  ✓  Secret PSNR acceptable for {meta2['type']} reconstruction.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Unique subfolder per run: cover_TYPE__secret_TYPE__YYYYMMDD_HHMMSS
    run_tag = f"cover_{meta1['type']}__secret_{meta2['type']}__{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {out_dir}")

    # Reconstruct & save cover
    ext1 = Path(meta1["path"]).suffix
    save_modality(final1, meta1, str(out_dir / f"cover_reconstructed{ext1}"))

    # Reconstruct & save secret
    ext2 = Path(meta2["path"]).suffix
    save_modality(final2, meta2, str(out_dir / f"secret_reconstructed{ext2}"))

    # Save weights (this IS the steganographic carrier)
    weights_path = str(out_dir / "siren_weights.pth")
    torch.save({
        "model_state": model.state_dict(),
        "in_dim":      in_dim,
        "out_dim_1":   out_dim1,
        "out_dim_2":   out_dim2,
        "hidden_dim":  cfg["hidden_dim"],
        "num_layers":  cfg["num_layers"],
        "omega_0":     cfg["omega_0"],
        "meta1":       meta1,
        "meta2":       meta2,
        "key_hash":    hashlib.sha256(str(args.key).encode()).hexdigest(),
    }, weights_path)
    print(f"  Weights saved → {weights_path}")

    # PSNR thresholds for verdict (text needs higher bar due to char sensitivity)
    def _verdict(psnr, modality):
        thresholds = {"text": (38, 30, 20), "audio": (35, 25, 15)}.get(modality, (35, 28, 18))
        good, ok, bad = thresholds
        if psnr >= good: return "GOOD — output should be clean"
        if psnr >= ok:   return "OK — minor errors likely"
        if psnr >= bad:  return "POOR — noticeable corruption"
        return "UNUSABLE — re-run with higher quality"

    # Save report
    report = {
        "mode":          "hide",
        "modal1":        {"type": meta1["type"], "file": meta1["path"],
                          "samples": int(data1.shape[0])},
        "modal2":        {"type": meta2["type"], "file": meta2["path"],
                          "samples": int(data2.shape[0])},
        "quality":       args.quality,
        "loss_weight_cover":  round(w_cover, 3),
        "loss_weight_secret": round(w_secret, 3),
        "steps":         cfg["steps"],
        "hidden_dim":    cfg["hidden_dim"],
        "total_params":  total_params,
        "psnr_cover_dB": round(psnr1, 4),
        "psnr_secret_dB":round(psnr2, 4),
        "verdict_cover":  _verdict(psnr1, meta1["type"]),
        "verdict_secret": _verdict(psnr2, meta2["type"]),
        "training_time_s": round(elapsed, 2),
        "device":        str(device),
        "key_hash":      hashlib.sha256(str(args.key).encode()).hexdigest(),
    }
    report_path = str(out_dir / "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {report_path}")
    print(f"\n{'═'*58}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  EXTRACT MODE
# ═════════════════════════════════════════════════════════════════════════════
def extract(args, device):
    print(f"\n{'═'*58}")
    print(f"  MODE: EXTRACT  |  Key: {args.key}")
    print(f"{'═'*58}\n")

    assert args.weights, "--weights path to siren_weights.pth is required for extract mode."
    ckpt = torch.load(args.weights, map_location=device)

    # Verify key
    key_hash = hashlib.sha256(str(args.key).encode()).hexdigest()
    if key_hash != ckpt["key_hash"]:
        print("  ✗ WRONG KEY — extraction will produce noise, not the secret.")
        print("    (Proceeding anyway for demonstration...)\n")

    meta1 = ckpt["meta1"]
    meta2 = ckpt["meta2"]

    model = CrossModalSIREN(
        in_dim     = ckpt["in_dim"],
        out_dim_1  = ckpt["out_dim_1"],
        out_dim_2  = ckpt["out_dim_2"],
        hidden_dim = ckpt["hidden_dim"],
        num_layers = ckpt["num_layers"],
        omega_0    = ckpt["omega_0"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Loaded model from: {args.weights}")
    print(f"  Cover  was: {meta1['type'].upper()}  ({meta1['path']})")
    print(f"  Secret was: {meta2['type'].upper()}  ({meta2['path']})")

    # Unique subfolder per extraction run
    run_tag = f"extract_cover_{meta1['type']}__secret_{meta2['type']}__{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {out_dir}")

    # ── Reconstruct cover (no key needed) ────────────────────────────────────
    _, coords1, _ = load_modality(meta1["path"], device, args.max_size)
    in_dim = ckpt["in_dim"]
    if coords1.shape[1] < in_dim:
        pad = torch.zeros(coords1.shape[0], in_dim - coords1.shape[1], device=device)
        coords1 = torch.cat([coords1, pad], dim=1)

    with torch.no_grad():
        cover_out = model.forward_cover(coords1)
    ext1 = Path(meta1["path"]).suffix
    save_modality(cover_out, meta1, str(out_dir / f"extracted_cover{ext1}"))

    # ── Reconstruct secret (key required) ────────────────────────────────────
    _, coords2, _ = load_modality(meta2["path"], device, args.max_size)
    if coords2.shape[1] < in_dim:
        pad = torch.zeros(coords2.shape[0], in_dim - coords2.shape[1], device=device)
        coords2 = torch.cat([coords2, pad], dim=1)

    shift        = key_to_shift(args.key, in_dim, device)
    coords2_keyed = apply_key(coords2, shift)

    with torch.no_grad():
        secret_out = model.forward_secret(coords2_keyed)
    ext2 = Path(meta2["path"]).suffix
    save_modality(secret_out, meta2, str(out_dir / f"extracted_secret{ext2}"))

    print(f"\n  ✓ Extraction complete")
    print(f"{'═'*58}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Cross-Modal INR Steganography",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--modal1",   type=str, default=None,
                        help="Cover modality file (image/audio/video/txt)")
    parser.add_argument("--modal2",   type=str, default=None,
                        help="Secret modality file (image/audio/video/txt)")
    parser.add_argument("--mode",     type=str, default="hide",
                        choices=["hide", "extract"],
                        help="hide: encode both into INR | extract: recover from weights")
    parser.add_argument("--key",      type=int, default=42,
                        help="Integer key for secret coordinate shift (default: 42)")
    parser.add_argument("--quality",  type=str, default=None,
                        choices=["fast", "low", "medium", "high", "ultra"],
                        help="Quality preset (fast/low/medium/high/ultra). Auto-selected if omitted.\n  fast: 1000 steps, dim 128 — for very slow CPUs, results likely poor\n  low:  3000 steps, dim 256 — minimum recommended\n  medium/high/ultra: GPU recommended")
    parser.add_argument("--weights",  type=str, default=None,
                        help="Path to siren_weights.pth (required for extract mode)")
    parser.add_argument("--output",   type=str, default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--max_size", type=int, default=128,
                        help="Max spatial resolution for image/video (default: 128)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Quality preset
    if args.quality is None:
        args.quality = auto_quality(device)
        print(f"\n  [Auto] Quality preset: {args.quality.upper()}  (device: {device})")
    cfg = QUALITY_PRESETS[args.quality]

    if args.mode == "hide":
        assert args.modal1 and args.modal2, \
            "Both --modal1 and --modal2 are required for hide mode."
        hide(args, device, cfg)
    elif args.mode == "extract":
        assert args.weights, "--weights is required for extract mode."
        assert args.modal1 or args.modal2, \
            "Provide --modal1 and/or --modal2 (original files) for coord reconstruction."
        extract(args, device)


if __name__ == "__main__":
    main()

#updated file
