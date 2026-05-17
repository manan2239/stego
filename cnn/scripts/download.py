import os
import requests
from concurrent.futures import ThreadPoolExecutor

TOTAL = 10000
SAVE_DIR = "../data/images"
MAX_WORKERS = 50   # tune this depending on internet speed

os.makedirs(SAVE_DIR, exist_ok=True)


def download_image(i):
    url = f"https://picsum.photos/512?random={i}"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code == 200:
            path = os.path.join(SAVE_DIR, f"img_{i}.jpg")

            with open(path, "wb") as f:
                f.write(r.content)

            print(f"saved {path}")

        else:
            print(f"failed {i} status={r.status_code}")

    except Exception as e:
        print(f"error downloading {i}: {e}")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(download_image, range(TOTAL))