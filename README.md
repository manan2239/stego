# LSB Text Steganography Toolkit

A simple but complete **Least Significant Bit (LSB)** steganography toolkit built in Python — designed to hide **text messages inside images** with optional compression, encryption, and bit-level control.

---

##  Features

- Hide **text messages** inside images (PNG recommended)
- Extract hidden text from stego images
- Optional **zlib compression** for compact payloads
- Optional **encryption** using XOR passphrases
- Configurable **bits per channel** and color channels (`R`, `G`, `B`)
- Deterministic **bit scattering** with a random seed
- CLI interface — no dependencies beyond `Pillow` and `NumPy`

---

##  Requirements

Python 3.8+
```bash
pip install -r requirements.txt
