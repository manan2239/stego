#!/usr/bin/env python3
"""
Audio-in-Image Steganography CLI.

Embeds an audio file inside one or more copies of an RGB(A) image by
manipulating the least-significant bits (LSB) of the pixel channels.
If the audio exceeds a single image's capacity, the payload is split
across multiple stego images automatically. Extraction reassembles all
parts by reading the metadata embedded in each image.

Usage:
    python stego.py embed <cover_image> <audio_file> <output_image>
    python stego.py extract <stego_image> <output_audio>

The embed command writes one image when possible or a series of numbered
images (e.g. output_part01.png, output_part02.png, ...) when splitting
is required. The extract command accepts any part from the series and
automatically locates and joins the remaining parts in the same directory.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from PIL import Image, UnidentifiedImageError  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install it with 'pip install Pillow'."
    ) from exc

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "cryptography is required. Install it with 'pip install cryptography'."
    ) from exc


MAGIC = b"STEG"
MAGIC_LEN = len(MAGIC)
VERSION = 2  # Incremented to support compression and encryption
GROUP_ID_LEN = 16
SIZE_BYTES = 4  # 32 bits for payload size
UINT16_MAX = 0xFFFF
AES_KEY_SIZE = 32  # 256 bits
AES_NONCE_SIZE = 12  # 96 bits for GCM
SALT_SIZE = 16
PBKDF2_ITERATIONS = 100000
HEADER_PREFIX_BYTES = MAGIC_LEN + 1 + GROUP_ID_LEN + 2 + 2 + SIZE_BYTES + SIZE_BYTES + 1


@dataclass(frozen=True)
class PayloadHeader:
    version: int
    group_id: bytes
    total_parts: int
    part_index: int
    audio_size: int
    chunk_size: int
    extension: str


class StegoError(Exception):
    """Custom exception for steganography errors."""


def compress_data(data: bytes) -> bytes:
    """Compress data using gzip."""
    return gzip.compress(data, compresslevel=9)


def decompress_data(compressed_data: bytes) -> bytes:
    """Decompress gzip-compressed data."""
    try:
        return gzip.decompress(compressed_data)
    except gzip.BadGzipFile as exc:
        raise StegoError("Failed to decompress data.") from exc


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive AES key from password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend(),
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_data(data: bytes, password: str) -> bytes:
    """Encrypt data using AES-256-GCM."""
    salt = os.urandom(SALT_SIZE)
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    nonce = os.urandom(AES_NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    # Prepend salt and nonce to ciphertext
    return salt + nonce + ciphertext


def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
    """Decrypt AES-256-GCM encrypted data."""
    if len(encrypted_data) < SALT_SIZE + AES_NONCE_SIZE:
        raise StegoError("Encrypted data is too short.")
    
    salt = encrypted_data[:SALT_SIZE]
    nonce = encrypted_data[SALT_SIZE:SALT_SIZE + AES_NONCE_SIZE]
    ciphertext = encrypted_data[SALT_SIZE + AES_NONCE_SIZE:]
    
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise StegoError("Decryption failed. Incorrect password?") from exc


def bytes_to_bits(data: bytes) -> Iterator[int]:
    """Yield individual bits (0/1) from a bytes object."""
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    """Convert an iterable of bits into bytes."""
    out = bytearray()
    byte = 0
    count = 0
    for bit in bits:
        byte = (byte << 1) | (bit & 1)
        count += 1
        if count == 8:
            out.append(byte)
            byte = 0
            count = 0
    if count != 0:
        raise StegoError("Bit stream length is not a multiple of 8.")
    return bytes(out)


def flatten_channels(pixels: Sequence[Sequence[int]]) -> list[int]:
    """Flatten a sequence of RGB tuples into a list of channel values."""
    flat: list[int] = []
    for pixel in pixels:
        flat.extend(pixel)
    return flat


def chunk_channels(channels: Sequence[int], channels_per_pixel: int = 3) -> list[tuple[int, ...]]:
    """Group a flat list of channels back into pixel tuples."""
    if len(channels) % channels_per_pixel != 0:
        raise StegoError("Channel list length is not divisible by channels per pixel.")
    pixels: list[tuple[int, ...]] = []
    for idx in range(0, len(channels), channels_per_pixel):
        pixels.append(tuple(channels[idx : idx + channels_per_pixel]))
    return pixels


def header_bit_length(extension: str) -> int:
    ext_bytes = extension.encode("utf-8")
    return (HEADER_PREFIX_BYTES + len(ext_bytes)) * 8


def build_header(
    group_id: bytes,
    total_parts: int,
    part_index: int,
    audio_size: int,
    chunk_size: int,
    extension: str,
) -> bytes:
    if len(group_id) != GROUP_ID_LEN:
        raise StegoError("Invalid group id length.")
    if not (1 <= total_parts <= UINT16_MAX):
        raise StegoError("Total parts must be between 1 and 65535.")
    if not (1 <= part_index <= total_parts):
        raise StegoError("Part index out of range.")
    if not (0 <= audio_size < 2**32):
        raise StegoError("Audio size must be < 4 GiB.")
    if not (0 <= chunk_size < 2**32):
        raise StegoError("Chunk size must be < 4 GiB.")

    ext = extension.lower()
    ext_bytes = ext.encode("utf-8")
    if len(ext_bytes) > 255:
        raise StegoError("Audio file extension is too long (max 255 bytes).")

    header = bytearray()
    header.extend(MAGIC)
    header.append(VERSION)
    header.extend(group_id)
    header.extend(total_parts.to_bytes(2, "big"))
    header.extend(part_index.to_bytes(2, "big"))
    header.extend(audio_size.to_bytes(SIZE_BYTES, "big"))
    header.extend(chunk_size.to_bytes(SIZE_BYTES, "big"))
    header.append(len(ext_bytes))
    header.extend(ext_bytes)
    return bytes(header)


def parse_header(bit_iter: Iterator[int]) -> PayloadHeader:
    prefix_bits = HEADER_PREFIX_BYTES * 8
    header_bytes = bits_to_bytes(next_bits(bit_iter, prefix_bits))

    magic = header_bytes[:MAGIC_LEN]
    if magic != MAGIC:
        raise StegoError("Magic bytes not found; this image may not contain embedded audio.")

    version = header_bytes[MAGIC_LEN]
    if version not in (1, VERSION):
        raise StegoError(f"Unsupported payload version: {version}.")

    offset = MAGIC_LEN + 1
    group_id = header_bytes[offset : offset + GROUP_ID_LEN]
    offset += GROUP_ID_LEN

    total_parts = int.from_bytes(header_bytes[offset : offset + 2], "big")
    offset += 2
    part_index = int.from_bytes(header_bytes[offset : offset + 2], "big")
    offset += 2
    audio_size = int.from_bytes(header_bytes[offset : offset + SIZE_BYTES], "big")
    offset += SIZE_BYTES
    chunk_size = int.from_bytes(header_bytes[offset : offset + SIZE_BYTES], "big")
    offset += SIZE_BYTES
    ext_len = header_bytes[offset]

    if not (1 <= total_parts <= UINT16_MAX):
        raise StegoError("Invalid total parts in header.")
    if not (1 <= part_index <= total_parts):
        raise StegoError("Invalid part index in header.")
    if audio_size >= 2**32:
        raise StegoError("Encoded audio size exceeds supported limit.")
    if chunk_size >= 2**32:
        raise StegoError("Encoded chunk size exceeds supported limit.")

    ext_bytes = bits_to_bytes(next_bits(bit_iter, ext_len * 8)) if ext_len else b""
    try:
        extension = ext_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StegoError("Failed to decode audio extension from header.") from exc

    return PayloadHeader(
        version=version,
        group_id=group_id,
        total_parts=total_parts,
        part_index=part_index,
        audio_size=audio_size,
        chunk_size=chunk_size,
        extension=extension,
    )


def next_bits(bit_iter: Iterator[int], count: int) -> Iterator[int]:
    """Take a fixed number of bits from an iterator."""
    for _ in range(count):
        try:
            yield next(bit_iter)
        except StopIteration as exc:
            raise StegoError("Unexpected end of bit stream.") from exc


def part_filename(base_path: Path, part_index: int, total_parts: int) -> Path:
    if total_parts == 1:
        return base_path

    suffix = "".join(base_path.suffixes)
    name = base_path.name
    if suffix:
        name = name[: -len(suffix)]
    width = max(2, len(str(total_parts)))
    new_name = f"{name}_part{part_index:0{width}d}{suffix}"
    return base_path.with_name(new_name)


def embed_audio(cover_path: Path, audio_path: Path, output_path: Path, password: Optional[str] = None) -> List[Path]:
    """Embed audio file into one or more stego images and return their paths."""
    with Image.open(cover_path) as img:
        if img.mode not in {"RGB", "RGBA"}:
            img = img.convert("RGB")
        mode = img.mode
        channels_per_pixel = len(mode)
        image_size = img.size
        pixels = list(img.getdata())

    flat_channels = flatten_channels(pixels)
    capacity_bits = len(flat_channels)

    extension = audio_path.suffix.lower().lstrip(".")
    header_bits = header_bit_length(extension)
    usable_bits = capacity_bits - header_bits
    if usable_bits < 8:
        raise StegoError("Cover image is too small to store the stego header.")

    max_chunk_bytes = usable_bits // 8
    if max_chunk_bytes <= 0:
        raise StegoError("Cover image cannot store any audio data.")

    audio_bytes = audio_path.read_bytes()
    if len(audio_bytes) >= 2**32:
        raise StegoError("Audio file is too large (>= 4 GiB).")

    # Compress the audio data
    compressed_bytes = compress_data(audio_bytes)
    
    # Encrypt if password is provided
    if password:
        processed_bytes = encrypt_data(compressed_bytes, password)
    else:
        processed_bytes = compressed_bytes

    original_size = len(audio_bytes)
    processed_size = len(processed_bytes)

    if processed_size == 0:
        total_parts = 1
    else:
        total_parts = math.ceil(processed_size / max_chunk_bytes)

    if total_parts > UINT16_MAX:
        raise StegoError(
            "Audio requires more than 65,535 parts; use a larger cover image or compress the audio."
        )

    group_id = os.urandom(GROUP_ID_LEN)
    outputs: List[Path] = []
    offset = 0

    for part_index in range(1, total_parts + 1):
        chunk = processed_bytes[offset : offset + max_chunk_bytes]
        offset += len(chunk)

        header_bytes = build_header(
            group_id=group_id,
            total_parts=total_parts,
            part_index=part_index,
            audio_size=processed_size,  # Store processed size (compressed+encrypted)
            chunk_size=len(chunk),
            extension=extension,
        )
        payload_bytes = header_bytes + chunk
        payload_bits = list(bytes_to_bits(payload_bytes))

        if len(payload_bits) > capacity_bits:
            raise StegoError("Internal error: payload exceeds cover capacity.")

        embedded_channels = flat_channels[:]
        for idx, bit in enumerate(payload_bits):
            embedded_channels[idx] = (embedded_channels[idx] & ~1) | bit

        pixels_embedded = chunk_channels(embedded_channels, channels_per_pixel)
        stego_img = Image.new(mode, image_size)
        stego_img.putdata(pixels_embedded)

        out_path = part_filename(output_path, part_index, total_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stego_img.save(out_path)
        outputs.append(out_path)

    if offset != processed_size:
        raise StegoError("Failed to embed the complete audio payload.")

    return outputs


def read_stego_part(stego_path: Path) -> Tuple[PayloadHeader, bytes]:
    with Image.open(stego_path) as img:
        if img.mode not in {"RGB", "RGBA"}:
            raise StegoError("Unsupported image mode for extraction.")
        pixels = list(img.getdata())

    flat_channels = flatten_channels(pixels)
    bit_stream = ((channel & 1) for channel in flat_channels)

    header = parse_header(bit_stream)
    chunk_bits = next_bits(bit_stream, header.chunk_size * 8)
    chunk_bytes = bits_to_bytes(chunk_bits)

    if len(chunk_bytes) != header.chunk_size:
        raise StegoError("Chunk size mismatch while extracting audio.")

    return header, chunk_bytes


def extract_audio(stego_path: Path, output_path: Path, password: Optional[str] = None) -> Tuple[Path, int]:
    """Extract embedded audio from one or multiple stego images."""
    header, chunk = read_stego_part(stego_path)
    parts: Dict[int, Tuple[PayloadHeader, bytes]] = {header.part_index: (header, chunk)}

    if header.total_parts > 1:
        directory = stego_path.parent
        for candidate in sorted(directory.iterdir()):
            if candidate == stego_path or candidate.is_dir():
                continue
            try:
                candidate_header, candidate_chunk = read_stego_part(candidate)
            except (FileNotFoundError, StegoError, UnidentifiedImageError, OSError):
                continue

            if candidate_header.group_id != header.group_id:
                continue
            if (
                candidate_header.total_parts != header.total_parts
                or candidate_header.audio_size != header.audio_size
                or candidate_header.extension.lower() != header.extension.lower()
            ):
                continue

            parts[candidate_header.part_index] = (candidate_header, candidate_chunk)
            if len(parts) == header.total_parts:
                break

        if len(parts) != header.total_parts:
            missing = sorted(
                set(range(1, header.total_parts + 1)) - set(parts.keys())
            )
            missing_str = ", ".join(str(m) for m in missing)
            raise StegoError(
                f"Missing stego image(s) for part(s): {missing_str}."
            )

    ordered_chunks: List[bytes] = []
    total_collected = 0
    for idx in range(1, header.total_parts + 1):
        part_header, part_chunk = parts[idx]
        if len(part_chunk) != part_header.chunk_size:
            raise StegoError(f"Extracted chunk size mismatch for part {idx}.")
        ordered_chunks.append(part_chunk)
        total_collected += len(part_chunk)

    processed_bytes = b"".join(ordered_chunks)
    if total_collected != header.audio_size:
        raise StegoError("Reconstructed audio size does not match metadata.")

    # Handle version 1 (no compression/encryption) vs version 2+ (compressed/encrypted)
    if header.version == 1:
        # Version 1: no compression or encryption
        audio_bytes = processed_bytes
    else:
        # Version 2+: compressed and optionally encrypted
        # Decrypt if password is provided
        if password:
            try:
                compressed_bytes = decrypt_data(processed_bytes, password)
            except StegoError:
                raise StegoError("Decryption failed. Incorrect password or data corrupted.")
        else:
            # Try to decrypt anyway (might be encrypted)
            # If it fails, assume it's not encrypted
            try:
                compressed_bytes = decrypt_data(processed_bytes, "")
            except StegoError:
                compressed_bytes = processed_bytes
        
        # Decompress the data
        try:
            audio_bytes = decompress_data(compressed_bytes)
        except StegoError:
            raise StegoError("Failed to decompress data. File may be corrupted.")

    output_file = output_path
    if not output_file.suffix and header.extension:
        output_file = output_file.with_suffix("." + header.extension)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(audio_bytes)
    return output_file, header.total_parts


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hide audio inside an image using LSB steganography.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Embed audio in an image.")
    embed_parser.add_argument("cover_image", type=Path, help="Path to the cover image.")
    embed_parser.add_argument("audio_file", type=Path, help="Path to the audio file.")
    embed_parser.add_argument(
        "output_image",
        type=Path,
        help="Path for the stego image output.",
    )
    embed_parser.add_argument(
        "--password",
        "-p",
        type=str,
        help="Password for AES encryption (optional but recommended).",
    )

    extract_parser = subparsers.add_parser("extract", help="Extract audio from an image.")
    extract_parser.add_argument("stego_image", type=Path, help="Image containing hidden audio.")
    extract_parser.add_argument(
        "output_audio",
        type=Path,
        help="Path for the extracted audio (extension optional).",
    )
    extract_parser.add_argument(
        "--password",
        "-p",
        type=str,
        help="Password for AES decryption (required if encryption was used).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    try:
        if args.command == "embed":
            outputs = embed_audio(args.cover_image, args.audio_file, args.output_image, args.password)
            audio_name = args.audio_file
            if len(outputs) == 1:
                print(f"Embedded '{audio_name}' into '{outputs[0]}'.")
            else:
                print(
                    f"Embedded '{audio_name}' into {len(outputs)} images:"
                )
                for path in outputs:
                    print(f"  {path}")
        elif args.command == "extract":
            output_file, part_count = extract_audio(args.stego_image, args.output_audio, args.password)
            if part_count == 1:
                print(f"Extracted audio to '{output_file}'.")
            else:
                print(
                    f"Extracted audio (from {part_count} images) to '{output_file}'."
                )
        else:  # pragma: no cover
            raise StegoError(f"Unknown command: {args.command}")
    except (FileNotFoundError, StegoError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

