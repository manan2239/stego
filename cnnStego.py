#!/usr/bin/env python3
"""
Image-in-image steganography with automatic GPU/CPU switching.
Uses a lightweight CNN that can run on low-end CPUs,
but automatically accelerates using CUDA if available.

Usage:
TRAIN:
    python cnnStego.py train 
    --data-dir ./data/images 
    --epochs 10 
    --batch-size 8 
    --image-size 128 
    --checkpoint-dir ./checkpoints

TEST:
    python cnnStego.py test 
    --data-dir ./data/images 
    --checkpoint ./checkpoints/epoch_10.pth 
    --output-dir ./outputs 
    --num-samples 5
"""
import os
import math
import random
import argparse
import sys
from typing import Tuple
from PIL import Image
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except Exception as e:
    print("Failed to import PyTorch or related packages. Make sure the selected Python interpreter has 'torch' installed.", file=sys.stderr)
    raise

#Dataset
class ImagePairDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int=128):
        super().__init__()
        if not os.path.isdir(root_dir):
            raise RuntimeError(f"Data directory not found: {root_dir}")
        self.root_dir=root_dir
        valid_ext={".png",".jpg",".jpeg",".bmp"}
        self.image_paths=sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f.lower())[1] in valid_ext
        ])
        if len(self.image_paths) < 2:
            raise RuntimeError("Dataset must contain at least 2 images.")
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx:int):
        cover_path=self.image_paths[idx]

        secret_idx=random.randint(0, len(self.image_paths)-2)
        if secret_idx >= idx:
            secret_idx+=1

        secret_path=self.image_paths[secret_idx]

        cover=Image.open(cover_path).convert("RGB")
        secret=Image.open(secret_path).convert("RGB")

        return self.transform(cover), self.transform(secret)
    
#Models
class Encoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels*2, 3, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, cover, secret):
        x=torch.cat([cover, secret], dim=1)
        return self.net(x)
    
class Decoder(nn.Module):
    def  __init__(self, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(base_channels*2, 3, 1),
            nn.Sigmoid(),
        )
    def forward(self, stego):
        return self.net(stego)
    
#Utils
def psnr(x, y):
    mse=torch.mean((x-y)**2).item()
    return 99.0 if mse==0 else 10*math.log10(1.0/mse)

def save_tensor_image(tensor, path):
    tensor=tensor.detach().cpu().clamp(0,1)
    img=T.ToPILImage()(tensor)
    outdir = os.path.dirname(path) or "."
    os.makedirs(outdir, exist_ok=True)
    img.save(path)

#Training
def train(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dataset=ImagePairDataset(args.data_dir, args.image_size)
    # don't drop last batch to avoid empty dataloader when dataset < batch_size
    dataloader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    encoder=Encoder(args.base_channels).to(device)
    decoder=Decoder(args.base_channels).to(device)

    optimizer=optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    loss_fn=nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        encoder.train()
        decoder.train()

        epoch_loss=0

        for cover, secret in dataloader:
            cover, secret = cover.to(device), secret.to(device)

            stego=encoder(cover, secret)
            recovered=decoder(stego)

            loss_cover=loss_fn(stego, cover)
            loss_secret=loss_fn(recovered, secret)

            loss=args.cover_weight*loss_cover+args.secret_weight*loss_secret

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

        n_batches = len(dataloader) if len(dataloader) > 0 else 1
        print(f"Epoch {epoch}/{args.epochs} | Loss: {epoch_loss/n_batches:.6f}")

        ckpt_path=os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "args": vars(args),
        }, ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}")

#Testing
def test(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    dataset=ImagePairDataset(args.data_dir, args.image_size)

    encoder= Encoder(args.base_channels).to(device)
    decoder= Decoder(args.base_channels).to(device)

    ckpt=torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(min(args.num_samples, len(dataset))):
        cover, secret = dataset[i]
        cover_batch=cover.unsqueeze(0).to(device)
        secret_batch=secret.unsqueeze(0).to(device)

        with torch.no_grad():
            stego=encoder(cover_batch, secret_batch)[0]
            recovered=decoder(stego.unsqueeze(0))[0]

        print(f"Sample {i} | PSNR Cover->Stego: {psnr(stego, cover):.2f} dB"
              f"| PSNR Secret->Recovered: {psnr(recovered, secret):.2f} dB")
        
        save_tensor_image(cover, os.path.join(args.output_dir, f"{i}_cover.png"))
        save_tensor_image(secret, os.path.join(args.output_dir, f"{i}_secret.png"))
        save_tensor_image(stego, os.path.join(args.output_dir, f"{i}_stego.png"))
        save_tensor_image(recovered, os.path.join(args.output_dir, f"{i}_recovered.png"))

    print(f"Outputs saved at {args.output_dir}")

#Main
def main():
    parser=argparse.ArgumentParser(description="GPU/CPU Image Steganography")
    sub=parser.add_subparsers(dest="mode", required=True)

    #Train
    t=sub.add_parser("train")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--image-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--cover-weight", type=float, default=1.0)
    t.add_argument("--secret-weight", type=float, default=1.0)
    t.add_argument("--base-channels", type=int, default=32)
    t.add_argument("--checkpoint-dir", default="./checkpoints")

    # Test
    te = sub.add_parser("test")
    te.add_argument("--data-dir", required=True)
    te.add_argument("--checkpoint", required=True)
    te.add_argument("--output-dir", default="./outputs")
    te.add_argument("--image-size", type=int, default=128)
    te.add_argument("--base-channels", type=int, default=32)
    te.add_argument("--num-samples", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()