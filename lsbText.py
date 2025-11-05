"""
LSB Text-in-Image Steganography Tool
Features:
-Embed plain text or a text file into an image and extract it back.
-Options: bits-per-channel (1-4), channel selection (R,G,B,A), compression(zlib), passphrase-based XOR obfuscation, and PRNG-based bit scattering.
-Capacity check (to ensure payload fits in cover image).

Dependencies:
- Python 3.x
- numpy
- Pillow (PIL)
Install: pip install numpy Pillow
"""

import argparse
import os
import struct
import hashlib
import zlib
import numpy as np
from PIL import Image

MAGIC = b"LSBTXT1"

def derive_keystream(key_bytes: bytes, n: int):
    """Simple SHA256-based keystream to XOR payload"""
    out=bytearray()
    ctr=0
    while len(out)<n:
        h=hashlib.sha256()
        h.update(key_bytes)
        h.update(struct.pack("!I", ctr))
        out.extend(h.digest())
        ctr+=1
    return bytes(out[:n])

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr=np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits=np.array(bits, dtype=np.uint8)
    extra=(-bits.size) % 8
    if extra:
        bits=np.concatenate([bits, np.zeros(extra, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

def estimate_capacity(img_shape, bits_per_channel, channels):
    h,w=img_shape[:2]
    chans=len(channels)
    return h*w*chans*bits_per_channel

def build_header(flags: int, payload_len: int, bits_per_channel: int, chan_mask: int):
    return MAGIC+struct.pack("!BIBB", flags, payload_len, bits_per_channel, chan_mask)

def parse_header(data: bytes):
    # expected header size = len(MAGIC) + 1(flags) + 4(payload_len) + 1(bits) + 1(chan_mask) = 14
    if len(data) < 7+1+4+1+1:
        raise ValueError("Header too short")
    if data[:7] != MAGIC:
        raise ValueError("MAGIC mismatch")
    ptr=7
    flags=data[ptr]; ptr+=1
    payload_len=struct.unpack("!I", data[ptr:ptr+4])[0]; ptr+=4
    bits_per_channel=data[ptr]; ptr+=1
    chan_mask=data[ptr]; ptr+=1
    return flags, payload_len, bits_per_channel, chan_mask, ptr

def chan_mask_from_string(ch_str: str):
    ch_str=ch_str.upper()
    mask=0
    cmap={'R':1, 'G':2, 'B':4, 'A':8}
    for c in ch_str:
        if c in cmap:
            mask |= cmap[c]
    return mask

def mask_to_channel_list(mask: int):
    order = []
    if mask & 1: order.append('R')
    if mask & 2: order.append('G') 
    if mask & 4: order.append('B')
    if mask & 8: order.append('A')
    return order

def embed_text(cover_path, text_or_file, out_path, bits=1, channels='B', compress=False, passphrase=None, seed=None):
    #read image
    img=Image.open(cover_path).convert('RGBA')
    arr=np.array(img)
    h,w=arr.shape[:2]

    #get text bytes
    if os.path.isfile(text_or_file):
        with open(text_or_file, 'rb') as f:
            payload=f.read()
    else:
        payload=text_or_file.encode('utf-8')
    
    flags=0
    if compress:
        payload=zlib.compress(payload)
        flags |= 0x1
    if passphrase:
        key=hashlib.sha256(passphrase.encode('utf-8')).digest()
        ks=derive_keystream(key, len(payload))
        payload=bytes([p^k for p,k in zip(payload, ks)])
        flags |= 0x2

    header=build_header(flags, len(payload), bits, chan_mask_from_string(channels))
    full=header+payload
    bits_stream=bytes_to_bits(full)
    total_bits=bits_stream.size

    # prepare coordinates (row-major), with optional permutation
    coords=[(i,j) for i in range(h) for j in range(w)]
    if seed is not None:
        rng=np.random.default_rng(seed)
        coords=list(rng.permutation(coords))

    # channel indexes in same order as chan_list for payload
    idx_map={'R':0, 'G':1, 'B':2, 'A':3}
    chan_list=[c for c in channels.upper() if c in 'RGBA']
    ch_indices=[idx_map[c] for c in chan_list]

    # Build positions exactly as will be used for embedding:
    # - first embed the header using LSB (bitplane 0) of ALL channels in RGBA order
    # - then embed payload bits using the requested channels and bitplanes
    header_bits = len(header) * 8
    header_positions = []
    for (i,j) in coords:
        for ch in [0,1,2,3]:
            header_positions.append((i,j,ch,0))
            if len(header_positions) >= header_bits:
                break
        if len(header_positions) >= header_bits:
            break
    header_set = set(header_positions)

    # now build payload positions skipping any that were already used for header
    positions = list(header_positions)  # start with header positions
    for (i,j) in coords:
        for ch in ch_indices:
            for bitplane in range(bits):
                pos = (i,j,ch,bitplane)
                if pos in header_set:
                    # skip positions already used by header
                    continue
                positions.append(pos)
                if len(positions) >= total_bits:
                    break
            if len(positions) >= total_bits:
                break
        if len(positions) >= total_bits:
            break

    capacity = len(positions)
    if total_bits > capacity:
        raise ValueError(f"Payload requires {total_bits} bits but capacity is {capacity} bits. Reduce payload or increase bits/channels.")

    # perform embedding
    stego=arr.copy()
    for k,bit in enumerate(bits_stream):
        i,j,ch,bitplane = positions[k]
        mask=1 << bitplane
        inv_mask = np.uint8(0xFF ^ mask)            # ensure 8-bit inverse mask
        stego[i,j,ch] = np.uint8(stego[i,j,ch] & inv_mask)
        if bit:
            stego[i,j,ch] = np.uint8(stego[i,j,ch] | np.uint8(mask))

    out_img=Image.fromarray(stego, mode='RGBA' if img.mode=='RGBA' else 'RGB')
    out_img.save(out_path)
    return {'out_path':out_path, 'payload_bytes':len(payload), 'total_bits':total_bits, 'capacity_bits':capacity}

def extract_text(stego_path, passphrase=None, seed=None):
    img=Image.open(stego_path).convert('RGBA')
    arr=np.array(img)
    h,w=arr.shape[:2]

    coords=[(i,j) for i in range(h) for j in range(w)]
    if seed is not None:
        rng=np.random.default_rng(seed)
        coords=list(rng.permutation(coords))

    # read header bits: header is embedded in LSB (bitplane 0) of ALL channels in RGBA order
    header_bytes_needed=14
    header_bits_needed=header_bytes_needed*8

    bits_acc=[]
    for(i,j) in coords:
        for ch in [0,1,2,3]:
            bits_acc.append(arr[i,j,ch] & 1)
            if len(bits_acc) >= header_bits_needed:
                break
        if len(bits_acc) >= header_bits_needed:
            break

    header_bytes=bits_to_bytes(np.array(bits_acc[:header_bits_needed], dtype=np.uint8))
    if header_bytes[:7] != MAGIC:
        raise ValueError("MAGIC header not found. Wrong embedding params(bits/channels/seed) or not a stego image from this tool.")
    
    flags, payload_len, bits_used, chan_mask, hdr_end=parse_header(header_bytes)
    chan_list=mask_to_channel_list(chan_mask)
    idx_map={'R':0, 'G':1, 'B':2, 'A':3}
    ch_indices=[idx_map[c] for c in chan_list]

    total_payload_bits=(hdr_end+payload_len)*8

    # reconstruct positions the same way embedding did:
    # header positions first (LSB of all RGBA channels), then payload positions using chan_list and bits_used
    header_bits = hdr_end * 8
    header_positions = []
    for (i,j) in coords:
        for ch in [0,1,2,3]:
            header_positions.append((i,j,ch,0))
            if len(header_positions) >= header_bits:
                break
        if len(header_positions) >= header_bits:
            break
    header_set = set(header_positions)

    positions = list(header_positions)
    for (i,j) in coords:
        for ch in ch_indices:
            for bitplane in range(bits_used):
                pos = (i,j,ch,bitplane)
                if pos in header_set:
                    continue
                positions.append(pos)
                if len(positions) >= total_payload_bits:
                    break
            if len(positions) >= total_payload_bits:
                break
        if len(positions) >= total_payload_bits:
            break

    bits=[]
    for(i,j,ch,bitplane) in positions[:total_payload_bits]:
        bits.append((arr[i,j,ch] >> bitplane) & 1)

    data=bits_to_bytes(np.array(bits, dtype=np.uint8))
    if data[:7] != MAGIC:
        raise ValueError("MAGIC mismatch after reading full payload.")
    flags, payload_len, bits_used, chan_mask, hdr_end = parse_header(data)
    payload=data[hdr_end:hdr_end+payload_len]

    if flags & 0x2:
        if not passphrase:
            raise ValueError("Payload encrypted; passphrase required.")
        key=hashlib.sha256(passphrase.encode('utf-8')).digest()
        ks=derive_keystream(key, len(payload))
        payload=bytes([p^k for p,k in zip(payload, ks)])
    if flags & 0x1:
        payload=zlib.decompress(payload)

    #return text
    try:
        return payload.decode('utf-8')
    except Exception:
        return payload
    
def main():
    p = argparse.ArgumentParser(description='LSB Text-in-Image Stego')
    sub = p.add_subparsers(dest='cmd')

    p_embed = sub.add_parser('embed')
    p_embed.add_argument('--cover', required=True)
    p_embed.add_argument('--text', required=True, help='Either literal text or path to a text file')
    p_embed.add_argument('--out', required=True)
    p_embed.add_argument('--bits', type=int, default=1, choices=[1,2,3,4])
    p_embed.add_argument('--channels', default='B', help='Channels to use, e.g. B or RGB')
    p_embed.add_argument('--compress', action='store_true')
    p_embed.add_argument('--pass', dest='passphrase', default=None, help='Passphrase to XOR-obfuscate payload')
    p_embed.add_argument('--seed', type=int, default=None, help='PRNG seed to scatter bits')

    p_extract = sub.add_parser('extract')
    p_extract.add_argument('--stego', required=True)
    p_extract.add_argument('--pass', dest='passphrase', default=None)
    p_extract.add_argument('--seed', type=int, default=None)

    args=p.parse_args()
    if args.cmd == 'embed':
        info=embed_text(args.cover, args.text, args.out, bits=args.bits, channels=args.channels, 
                        compress=args.compress, passphrase=args.passphrase, seed=args.seed)
        print("Embed complete:")
        for k,v in info.items():
            print(f"{k}: {v}")
    elif args.cmd == 'extract':
        out=extract_text(args.stego, passphrase=args.passphrase, seed=args.seed)
        if isinstance(out, str):
            print("Extracted text:")
            print(out)
        else:
            #bytes returned
            fname="extracted_payload.bin"
            with open(fname, 'wb') as f:
                f.write(out)
            print(f"Binary payload saved to {fname}")
    else:
        p.print_help()

if __name__ == '__main__':
    main()

"""
Bash:
Embed literal text into image(blue channel, 1 LSB): python lsbText.py embed --cover (coverImage) --text "Secret Message" --out (outImage) 
                                                    --bits 1 --channels B

Embed a text file with compressionand passphrase+seed: python lsbText.py embed --cover (coverImage) --text (secretMessage) --out (outImage) --bits 1
                                                       --channels B --compress --pass mypass --seed 42

Extract: python lsbText.py extract --stego (outImage) --pass mypass --seed 42
"""