import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from pytorch_msssim import ssim


# Dataset
class StegoDataset(Dataset):

    def __init__(self, cover_dir, secret_dir, size=128):

        self.cover_dir = cover_dir
        self.secret_dir = secret_dir

        self.cover_imgs = sorted(os.listdir(cover_dir))
        self.secret_imgs = sorted(os.listdir(secret_dir))

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return min(len(self.cover_imgs), len(self.secret_imgs))

    def __getitem__(self, idx):

        cover = Image.open(
            os.path.join(self.cover_dir, self.cover_imgs[idx])
        ).convert("RGB")

        secret = Image.open(
            os.path.join(self.secret_dir, self.secret_imgs[idx])
        ).convert("RGB")

        return self.transform(cover), self.transform(secret)


# CNN Blocks
class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


# Encoder
class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = ConvBlock(6, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, cover, secret):

        x = torch.cat([cover, secret], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.d3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))


# Decoder
class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 64),

            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Perceptual Loss
class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg

    def forward(self, x, y):

        return F.mse_loss(self.vgg(x), self.vgg(y))


# Stego Loss
def stego_loss(stego, cover, recovered, secret):

    cover_mse = F.mse_loss(stego, cover)

    cover_ssim = 1 - ssim(
        stego,
        cover,
        data_range=1.0,
        size_average=True
    )

    secret_l1 = F.l1_loss(recovered, secret)

    return (
        1.0 * cover_mse +
        0.8 * cover_ssim +
        0.75 * secret_l1
    )


# Image helpers
def load_image(path, size=128):

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    img = Image.open(path).convert("RGB")

    return transform(img).unsqueeze(0)


def save_image(tensor, path):

    img = tensor.squeeze(0).detach().cpu()

    transforms.ToPILImage()(img).save(path)


# Training
def train(cover_dir, secret_dir, epochs=40, batch_size=4, capacity=1.0):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = StegoDataset(cover_dir, secret_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    perceptual = PerceptualLoss().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )

    for epoch in range(epochs):

        for cover, secret in loader:

            cover = cover.to(device)
            secret = secret.to(device)

            secret_scaled = secret * capacity

            stego = encoder(cover, secret_scaled)
            recovered = decoder(stego)

            percept = perceptual(stego, cover)

            loss = (
                stego_loss(stego, cover, recovered, secret)
                + 0.2 * percept
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")

    print("Training complete.")


# Embed
def embed(cover_path, secret_path, out_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    encoder.eval()

    cover = load_image(cover_path).to(device)
    secret = load_image(secret_path).to(device)

    with torch.no_grad():
        stego = encoder(cover, secret)

    save_image(stego, out_path)

    print("Stego saved:", out_path)


# Extract
def extract(stego_path, out_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
    decoder.eval()

    stego = load_image(stego_path).to(device)

    with torch.no_grad():
        secret = decoder(stego)

    save_image(secret, out_path)

    print("Recovered secret saved:", out_path)


# CLI
def parse_args():

    parser = argparse.ArgumentParser("CNN Steganography Tool")

    sub = parser.add_subparsers(dest="cmd", required=True)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--cover", required=True)
    train_cmd.add_argument("--secret", required=True)
    train_cmd.add_argument("--epochs", type=int, default=40)
    train_cmd.add_argument("--capacity", type=float, default=1.0)

    embed_cmd = sub.add_parser("embed")
    embed_cmd.add_argument("--cover", required=True)
    embed_cmd.add_argument("--secret", required=True)
    embed_cmd.add_argument("--out", required=True)

    extract_cmd = sub.add_parser("extract")
    extract_cmd.add_argument("--stego", required=True)
    extract_cmd.add_argument("--out", required=True)

    return parser.parse_args()


# Main
if __name__ == "__main__":

    args = parse_args()

    if args.cmd == "train":

        train(
            args.cover,
            args.secret,
            args.epochs,
            capacity=args.capacity
        )

    elif args.cmd == "embed":

        embed(args.cover, args.secret, args.out)

    elif args.cmd == "extract":

        extract(args.stego, args.out)