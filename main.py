"""
Usage:
    python main.py <method> [method-specific args...]

Methods:
    lsb            Text-in-image LSB steganography      -> lsb/lsbText.py
    audio          Audio-in-image LSB steganography      -> lsb/audioStego.py
    cnn            CNN encoder/decoder steganography     -> cnn/cnnStego.py
    inr            Cross-modal INR (SIREN) steganography -> inr/inrCM.py
    steganalysis   Blind steganalysis on a stego image    -> steganalysis/steganalysis.py

Examples:
    python main.py lsb embed --cover cover.png --text "hello" --out stego.png
    python main.py lsb extract --stego stego.png

    python main.py cnn train --cover cover.png --secret secret.png --epochs 40
    python main.py cnn embed --cover cover.png --secret secret.png --out stego.png
    python main.py cnn extract --stego stego.png --out recovered.png

    python main.py inr --modal1 cover.png --modal2 secret.txt --mode hide --key 42 --quality medium

    python main.py audio embed cover.png secret.wav --out stego.png
    python main.py audio extract stego.png --out recovered.wav

    python main.py steganalysis --cover cover.png --stego stego.png
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Map each method name to the script that implements it.
METHODS = {
    "lsb": BASE_DIR / "lsb" / "lsbText.py",
    "audio": BASE_DIR / "lsb" / "audioStego.py",
    "cnn": BASE_DIR / "cnn" / "cnnStego.py",
    "inr": BASE_DIR / "inr" / "inrCM.py",
    "steganalysis": BASE_DIR / "steganalysis" / "steganalysis.py",
}


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)

    method = sys.argv[1]

    if method not in METHODS:
        print(f"Unknown method: '{method}'")
        print(f"Available methods: {', '.join(METHODS)}\n")
        print_usage()
        sys.exit(1)

    script = METHODS[method]
    if not script.exists():
        print(f"Error: expected script not found at {script}")
        print("Make sure you're running main.py from the root of the stego repo.")
        sys.exit(1)

    # Forward every remaining argument as-is to the target script's own CLI.
    forwarded_args = sys.argv[2:]
    cmd = [sys.executable, str(script)] + forwarded_args

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
