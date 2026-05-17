import os
import shutil
import random

src = "../data/images"
cover_dir = "../data/cover"
secret_dir = "../data/secret"

os.makedirs(cover_dir, exist_ok=True)
os.makedirs(secret_dir, exist_ok=True)

files = os.listdir(src)
random.shuffle(files)

half = len(files) // 2

cover = files[:half]
secret = files[half:]

for f in cover:
    shutil.copy(os.path.join(src, f), os.path.join(cover_dir, f))

for f in secret:
    shutil.copy(os.path.join(src, f), os.path.join(secret_dir, f))

print("Dataset ready.")
