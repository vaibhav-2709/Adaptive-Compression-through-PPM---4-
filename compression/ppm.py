# compression/ppm.py
# Deterministic PPM-C (Prediction by Partial Matching, escape=C) with a simple range coder.
# Works on bytes input (binary-safe). Returns and accepts bytes.

import collections
from typing import List, Tuple

EOF_SYMBOL = 256        # special end-of-file marker (not a byte)
ALPHABET_BYTES = 256    # 0..255
ALPHABET_SIZE = ALPHABET_BYTES + 1  # includes EOF

# -------------------------------
# Range Encoder / Decoder (bit-oriented)
# -------------------------------

class RangeEncoder:
    def __init__(self, precision=32):
        self.PRECISION = precision
        self.FULL = 1 << precision
        self.HALF = self.FULL >> 1
        self.QUARTER = self.HALF >> 1
        self.MASK = self.FULL - 1

        self.low = 0
        self.high = self.MASK
        self.pending = 0
        self.out_bits = []

    def _emit_bit(self, bit: int):
        self.out_bits.append(bit)
        while self.pending > 0:
            self.out_bits.append(1 - bit)
            self.pending -= 1

    def encode_symbol(self, low_count: int, high_count: int, total: int):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * high_count // total) - 1
        self.low = self.low + (rng * low_count // total)

        while True:
            if self.high < self.HALF:
                self._emit_bit(0)
                self.low = (self.low << 1) & self.MASK
                self.high = ((self.high << 1) & self.MASK) | 1
            elif self.low >= self.HALF:
                self._emit_bit(1)
                self.low = ((self.low - self.HALF) << 1) & self.MASK
                self.high = (((self.high - self.HALF) << 1) & self.MASK) | 1
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.pending += 1
                self.low = ((self.low - self.QUARTER) << 1) & self.MASK
                self.high = (((self.high - self.QUARTER) << 1) & self.MASK) | 1
            else:
                break

    def finish(self) -> bytes:
        # Finalize: emit one more bit plus padding
        if self.low < self.QUARTER:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        # pack bits into bytes
        out = bytearray()
        cur = 0
        bits = 0
        for b in self.out_bits:
            cur = (cur << 1) | (b & 1)
            bits += 1
            if bits == 8:
                out.append(cur)
                cur = 0
                bits = 0
        if bits > 0:
            out.append(cur << (8 - bits))
        return bytes(out)


class RangeDecoder:
    def __init__(self, data: bytes, precision=32):
        self.PRECISION = precision
        self.FULL = 1 << precision
        self.HALF = self.FULL >> 1
        self.QUARTER = self.HALF >> 1
        self.MASK = self.FULL - 1

        self.low = 0
        self.high = self.MASK
        self.code = 0
        self.bits = []
        for b in data:
            for i in range(8):
                self.bits.append((b >> (7 - i)) & 1)
        self.pos = 0
        # init code with PRECISION bits
        for _ in range(self.PRECISION):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self.pos >= len(self.bits):
            return 0
        v = self.bits[self.pos]
        self.pos += 1
        return v

    def get_target(self, total: int) -> int:
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // rng

    def remove_symbol(self, low_count: int, high_count: int, total: int):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * high_count // total) - 1
        self.low = self.low + (rng * low_count // total)
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.code -= self.QUARTER
            else:
                break
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) & self.MASK) | 1
            self.code = ((self.code << 1) & self.MASK) | self._read_bit()


# -------------------------------
# PPM model
# -------------------------------

class PPMModel:
    def __init__(self, order=4):
        self.order = order
        self.contexts = {}  # map bytes -> Counter

    def get_counts(self, ctx: bytes) -> dict:
        return dict(self.contexts.get(ctx, {}))

    def update(self, ctx: bytes, symbol: int):
        if ctx not in self.contexts:
            self.contexts[ctx] = collections.Counter()
        self.contexts[ctx][symbol] += 1


# -------------------------------
# helpers
# -------------------------------

def build_cum_freqs_from_counts(counts: dict, excluded: set, include_escape: bool) -> Tuple[List[int], List[int]]:
    symbols = [s for s in sorted(counts.keys()) if s not in excluded]
    cum_freqs = [0]
    total = 0
    for s in symbols:
        total += counts[s]
        cum_freqs.append(total)
    if include_escape:
        total += 1
        cum_freqs.append(total)
    return symbols, cum_freqs


# -------------------------------
# PPM-C compressor (public)
# -------------------------------

class PPMCompressor:
    def __init__(self, order: int = 4):
        self.order = max(1, int(order))

    def compress_bytes(self, data: bytes) -> bytes:
        enc = RangeEncoder(precision=32)
        model = PPMModel(order=self.order)
        history = bytearray()

        for b in data:
            ctx = bytes(history[-self.order:])
            self._encode_symbol(enc, model, ctx, b)
            for k in range(self.order, -1, -1):
                subctx = ctx[-k:] if k > 0 else b''
                model.update(subctx, b)
            history.append(b)

        # EOF symbol encoding
        ctx = bytes(history[-self.order:]) if len(history) > 0 else b''
        self._encode_symbol(enc, model, ctx, EOF_SYMBOL)
        return enc.finish()

    def _encode_symbol(self, enc: RangeEncoder, model: PPMModel, ctx: bytes, symbol: int):
        excluded = set()
        for k in range(self.order, -1, -1):
            subctx = ctx[-k:] if k > 0 else b''
            counts = model.get_counts(subctx)
            symbols, cum_freqs = build_cum_freqs_from_counts(counts, excluded, include_escape=True)
            if cum_freqs[-1] == 0:
                continue
            if symbol in symbols:
                idx = symbols.index(symbol)
                low, high = cum_freqs[idx], cum_freqs[idx + 1]
                enc.encode_symbol(low, high, cum_freqs[-1])
                return
            else:
                # escape
                esc_low, esc_high = cum_freqs[-2], cum_freqs[-1]
                enc.encode_symbol(esc_low, esc_high, cum_freqs[-1])
                excluded.update(symbols)
        # fallback: include all bytes + EOF
        available = [s for s in range(ALPHABET_SIZE) if s not in excluded]
        total = len(available)
        idx = available.index(symbol)
        enc.encode_symbol(idx, idx + 1, total)

    def decompress_bytes(self, blob: bytes) -> bytes:
        dec = RangeDecoder(blob, precision=32)
        model = PPMModel(order=self.order)
        history = bytearray()
        out = bytearray()
        while True:
            ctx = bytes(history[-self.order:]) if len(history) > 0 else b''
            symbol = self._decode_symbol(dec, model, ctx)
            if symbol == EOF_SYMBOL:
                break
            out.append(symbol)
            for k in range(self.order, -1, -1):
                subctx = ctx[-k:] if k > 0 else b''
                model.update(subctx, symbol)
            history.append(symbol)
        return bytes(out)

    def _decode_symbol(self, dec: RangeDecoder, model: PPMModel, ctx: bytes) -> int:
        excluded = set()
        for k in range(self.order, -1, -1):
            subctx = ctx[-k:] if k > 0 else b''
            counts = model.get_counts(subctx)
            symbols, cum_freqs = build_cum_freqs_from_counts(counts, excluded, include_escape=True)
            if cum_freqs[-1] == 0:
                continue
            total = cum_freqs[-1]
            target = dec.get_target(total)
            for i in range(len(cum_freqs) - 1):
                if target < cum_freqs[i + 1]:
                    if i < len(symbols):
                        low, high = cum_freqs[i], cum_freqs[i + 1]
                        dec.remove_symbol(low, high, total)
                        return symbols[i]
                    else:
                        # escape
                        low, high = cum_freqs[i], cum_freqs[i + 1]
                        dec.remove_symbol(low, high, total)
                        excluded.update(symbols)
                        break
        # fallback: include all bytes + EOF
        available = [s for s in range(ALPHABET_SIZE) if s not in excluded]
        total = len(available)
        target = dec.get_target(total)
        idx = target
        dec.remove_symbol(idx, idx + 1, total)
        return available[idx]

