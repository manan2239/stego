"""
INR Image Fitting using SIREN
==============================
Fits any image using an Implicit Neural Representation.
The network learns: f(x, y) → (R, G, B)

Usage:
    python inr_image_fit.py --image path/to/image.png --steps 2000

If no image is provided, a synthetic test pattern is used.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Try to import PIL; fall back gracefully ──────────────────────────────────
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ─────────────────────────────────────────────────────────────────────────────
#  1. SIREN Layer
#     Key idea: uses sin(ω₀ · Wx + b) as activation.
#     This preserves high-frequency details much better than ReLU.
# ─────────────────────────────────────────────────────────────────────────────
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                # Hidden layers: uniform in [-√(6/in)/ω₀, √(6/in)/ω₀]
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ─────────────────────────────────────────────────────────────────────────────
#  2. SIREN Network  (the INR)
#     Input:  (N, 2)  — normalized (x, y) coordinates in [-1, 1]
#     Output: (N, 3)  — RGB values in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
class SIREN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, omega_0=30.0):
        super().__init__()

        layers = [SirenLayer(2, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        
        self.net = nn.Sequential(*layers)
        # Final linear layer + sigmoid to clamp output to [0,1]
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, coords):
        x = self.net(coords)
        return self.final(x)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Coordinate Grid
#     Creates a (H×W, 2) tensor of normalized pixel coordinates.
# ─────────────────────────────────────────────────────────────────────────────
def make_coordinate_grid(H, W):
    """Returns (H*W, 2) grid of (x,y) coords normalized to [-1, 1]."""
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    # meshgrid gives (H, W) each
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)   # (H, W, 2)
    return coords.reshape(-1, 2)                       # (H*W, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Image Loading / Synthetic Fallback
# ─────────────────────────────────────────────────────────────────────────────
def load_image(path, max_size=128):
    """Load image → normalized float tensor (H, W, 3) in [0,1]."""
    if path and HAS_PIL and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        print(f"  Loaded image: {img.size[0]}×{img.size[1]} px")
        return torch.from_numpy(arr)
    else:
        # Synthetic: colorful gradient + checkerboard
        print("  No image provided — generating synthetic test pattern.")
        H = W = max_size
        y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
        x = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
        r = x
        g = y
        b = 0.5 + 0.5 * torch.sin(x * 10 + y * 10)
        # Add checkerboard overlay
        checker = ((x * 8).int() + (y * 8).int()) % 2
        b = b * 0.7 + checker.float() * 0.3
        return torch.stack([r, g, b], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  5. PSNR metric
# ─────────────────────────────────────────────────────────────────────────────
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Save image tensor as PNG
# ─────────────────────────────────────────────────────────────────────────────
def save_image(tensor_hwc, path):
    """Save (H, W, 3) float tensor [0,1] as PNG."""
    arr = (tensor_hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if HAS_PIL:
        Image.fromarray(arr).save(path)
        print(f"  Saved → {path}")
    else:
        # Save raw numpy array if PIL not available
        np.save(path.replace('.png', '.npy'), arr)
        print(f"  PIL not found — saved numpy array → {path.replace('.png', '.npy')}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  INR Image Fitting  |  SIREN Architecture")
    print(f"{'='*55}")
    print(f"  Device  : {device}")

    # ── Load image ──────────────────────────────────────────────────────────
    img = load_image(args.image, max_size=args.max_size)  # (H, W, 3)
    H, W, _ = img.shape
    print(f"  Resolution : {W}×{H}")

    # Ground truth pixel values flattened: (H*W, 3)
    pixels = img.reshape(-1, 3).to(device)

    # Coordinate grid: (H*W, 2)
    coords = make_coordinate_grid(H, W).to(device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SIREN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=30.0
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params : {total_params:,}")
    print(f"  Image pixels : {H*W:,}")
    print(f"  Compression  : {H*W*3 / total_params:.2f}× (pixels/params)")
    print(f"{'='*55}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"  {'Step':>6}  {'Loss':>10}  {'PSNR (dB)':>10}")
    print(f"  {'-'*30}")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(coords)           # (H*W, 3)
        loss = loss_fn(pred, pixels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % (args.steps // 10) == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                pred_eval = model(coords)
                p = psnr(pred_eval, pixels)
            print(f"  {step:>6}  {loss.item():>10.6f}  {p:>10.2f}")

    # ── Final reconstruction ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_pred = model(coords).reshape(H, W, 3)
        final_psnr = psnr(final_pred.reshape(-1, 3), pixels)

    print(f"\n  ✓ Training complete")
    print(f"  Final PSNR : {final_psnr:.2f} dB")
    print(f"  (>30 dB = good quality | >40 dB = excellent)\n")

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs("inr_output", exist_ok=True)
    save_image(img,         "inr_output/original.png")
    save_image(final_pred,  "inr_output/reconstructed.png")

    # Save model weights
    torch.save(model.state_dict(), "inr_output/siren_weights.pth")
    print(f"  Weights saved → inr_output/siren_weights.pth")
    print(f"\n  KEY INSIGHT:")
    print(f"  The image is now encoded entirely in {total_params:,} floats")
    print(f"  (the network weights). No pixels — just a function f(x,y)→RGB.")
    print(f"{'='*55}\n")

    return model, coords, img


# ─────────────────────────────────────────────────────────────────────────────
#  8. Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INR Image Fitting with SIREN")
    parser.add_argument("--image",      type=str,   default=None,
                        help="Path to input image (PNG/JPG). Omit for test pattern.")
    parser.add_argument("--steps",      type=int,   default=2000,
                        help="Training iterations (default: 2000)")
    parser.add_argument("--hidden_dim", type=int,   default=256,
                        help="Hidden layer width (default: 256)")
    parser.add_argument("--num_layers", type=int,   default=5,
                        help="Number of layers (default: 5)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_size",   type=int,   default=128,
                        help="Max image dimension — resize if larger (default: 128)")
    args = parser.parse_args()

    train(args)