import os
import requests

TOTAL = 200                    # can be adjusted as needed
SAVE_DIR = "../data/images"

os.makedirs(SAVE_DIR, exist_ok=True)

for i in range(TOTAL):
    url = f"https://picsum.photos/512?random={i}"
    r = requests.get(url)
    path = os.path.join(SAVE_DIR, f"img_{i}.jpg")
    with open(path, "wb") as f:
        f.write(r.content)

    print("saved", path)
