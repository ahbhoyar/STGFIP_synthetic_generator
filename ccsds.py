"""Layer 3: CCSDS/ECSS encoding and an independent decoder (Sec. IV-D).

Space Packet (CCSDS 133.0-B) -> TM Transfer Frame (132.0-B, CRC-16)
-> CADU (131.0-B, ASM + Reed-Solomon (255,223)).
"""
import struct
import numpy as np

ASM = bytes.fromhex("1ACFFC1D")
RS_N, RS_K = 255, 223
NROOTS = RS_N - RS_K          # 32 -> corrects up to 16 symbol errors
FRAME_LEN  = 892              # CCSDS: 4 x 223, the WHOLE transfer frame
FRAME_DATA = FRAME_LEN - 6 - 2  # 884 B data field (6 B primary hdr + 2 B FECW)
APIDS = {"EPS": 0x101, "AOCS": 0x102, "THM": 0x103, "PRP": 0x104,
         "OBC": 0x105, "COM": 0x106, "NAV": 0x107, "PLD": 0x108}


# ---------------------------------------------------------------- CRC-16
def crc16_ccitt(data, crc=0xFFFF):
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
    return crc


# ------------------------------------------------- GF(256) Reed-Solomon
class GF:
    def __init__(self, poly=0x11D):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= poly
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a, b):
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % 255]

    def inv(self, a):
        return self.exp[(255 - self.log[a]) % 255]


GFF = GF()


def _gen_poly(nroots):
    g = [1]
    for i in range(nroots):
        g = _poly_mul(g, [1, GFF.exp[i]])
    return g


def _poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                r[i + j] ^= GFF.mul(a, b)
    return r


GEN = _gen_poly(NROOTS)


def rs_encode(msg):
    """Systematic RS(255,223) parity for a 223-byte message."""
    out = list(msg) + [0] * NROOTS
    for i in range(len(msg)):
        c = out[i]
        if c:
            for j in range(1, len(GEN)):
                out[i + j] ^= GFF.mul(GEN[j], c)
    return bytes(msg) + bytes(out[len(msg):])


_EXP = np.array(GFF.exp, dtype=np.int32)
_LOG = np.array(GFF.log, dtype=np.int32)
_ALPHA = _EXP[np.arange(NROOTS)]          # alpha^i for each root
_LOG_ALPHA = _LOG[_ALPHA]


def rs_syndromes(cw):
    """Horner evaluation at all NROOTS roots at once (vectorised over roots)."""
    v = np.zeros(NROOTS, dtype=np.int32)
    for b in cw:
        nz = v != 0
        v[nz] = _EXP[(_LOG[v[nz]] + _LOG_ALPHA[nz]) % 255]
        v ^= int(b)
    return v.tolist()


def rs_check(cw):
    """True if the codeword is consistent (no detected error)."""
    return all(v == 0 for v in rs_syndromes(cw))


def rs_nerrors(cw):
    """Berlekamp-Massey error count (0 if clean, -1 if beyond capability)."""
    syn = rs_syndromes(cw)
    if all(v == 0 for v in syn):
        return 0
    C, B, L, m, b = [1], [1], 0, 1, 1
    for n in range(NROOTS):
        d = syn[n]
        for i in range(1, L + 1):
            d ^= GFF.mul(C[i], syn[n - i])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T = list(C)
            coef = GFF.div(d, b)
            C = C + [0] * (len(B) + m - len(C)) if len(B) + m > len(C) else C
            for i in range(len(B)):
                C[i + m] ^= GFF.mul(coef, B[i])
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            coef = GFF.div(d, b)
            if len(B) + m > len(C):
                C = C + [0] * (len(B) + m - len(C))
            for i in range(len(B)):
                C[i + m] ^= GFF.mul(coef, B[i])
            m += 1
    return L if L <= NROOTS // 2 else -1


# --------------------------------------------- CCSDS pseudo-randomiser
# h(x) = x^8 + x^7 + x^5 + x^3 + 1, all-ones initial state.
# Verified against the Blue Book's published first 40 bits: FF 48 0E C0 9A.
def _rand_seq(n):
    st, out = 0xFF, bytearray()
    for _ in range(n):
        b = 0
        for _ in range(8):
            o = (st >> 7) & 1
            fb = ((st >> 0) ^ (st >> 2) ^ (st >> 4) ^ (st >> 7)) & 1
            st = ((st << 1) | fb) & 0xFF
            b = (b << 1) | o
        out.append(b)
    return bytes(out)


_RAND_CACHE = {}


def randomise(data):
    """XOR with the CCSDS randomising sequence. Self-inverse."""
    n = len(data)
    if n not in _RAND_CACHE:
        _RAND_CACHE[n] = _rand_seq(n)
    seq = _RAND_CACHE[n]
    return bytes(a ^ b for a, b in zip(data, seq))


# ------------------------------------------------------- Space Packet
def space_packet(apid, seq, payload, obt):
    """6-byte primary header + PUS data field header + payload."""
    ver_type_apid = (0 << 13) | (0 << 12) | (1 << 11) | (apid & 0x7FF)
    seq_field = (0b11 << 14) | (seq & 0x3FFF)
    pus = struct.pack(">BBBI", 0x20, 3, 25, int(obt) & 0xFFFFFFFF)
    body = pus + payload
    return struct.pack(">HHH", ver_type_apid, seq_field, len(body) - 1) + body


def parse_space_packet(buf):
    if len(buf) < 6:
        return None
    h0, h1, ln = struct.unpack(">HHH", buf[:6])
    apid = h0 & 0x7FF
    seq = h1 & 0x3FFF
    total = 6 + ln + 1
    if len(buf) < total:
        return None
    return dict(apid=apid, seq=seq, body=buf[6:total], total=total)


# --------------------------------------------------- TM Transfer Frame
def tm_frame(scid, vcid, mc_cnt, vc_cnt, data, fhp=0):
    w0 = (0 << 14) | ((scid & 0x3FF) << 4) | ((vcid & 0x7) << 1) | 0
    hdr = struct.pack(">HBBH", w0, mc_cnt & 0xFF, vc_cnt & 0xFF,
                      (0 << 11) | (fhp & 0x7FF))
    body = data.ljust(FRAME_DATA, b"\x00")[:FRAME_DATA]
    frame = hdr + body
    return frame + struct.pack(">H", crc16_ccitt(frame))


def parse_tm_frame(fr):
    if len(fr) != 6 + FRAME_DATA + 2:
        return None
    ok = crc16_ccitt(fr[:-2]) == struct.unpack(">H", fr[-2:])[0]
    w0, mc, vc, fhpw = struct.unpack(">HBBH", fr[:6])
    return dict(crc_ok=ok, scid=(w0 >> 4) & 0x3FF, vcid=(w0 >> 1) & 0x7,
                mc_cnt=mc, vc_cnt=vc, fhp=fhpw & 0x7FF, data=fr[6:6+FRAME_DATA])


# ------------------------------------------------------------- CADU
def cadu(frame):
    """ASM + randomised codeblock. 4 + 4*255 = 1024 B, per CCSDS 131.0-B.

    Order follows CCSDS 131.0-B Sec. 9: Reed-Solomon encode, then randomise the
    codeblock; the attached sync marker is NOT randomised.
    """
    assert len(frame) == FRAME_LEN, f"frame must be {FRAME_LEN} B, got {len(frame)}"
    inter = [bytes(frame[j::4]) for j in range(4)]          # 4 x 223
    parity = [rs_encode(v)[RS_K:] for v in inter]           # 4 x 32
    par = bytearray(4 * NROOTS)
    for i in range(NROOTS):
        for j in range(4):
            par[i * 4 + j] = parity[j][i]
    codeblock = bytes(frame) + bytes(par)                   # 892 + 128 = 1020
    return ASM + randomise(codeblock)


def decadu(buf):
    """Independent decode: ASM lock, de-randomise, RS check per interleave."""
    if buf[:4] != ASM:
        return dict(asm=False)
    codeblock = randomise(buf[4:4 + FRAME_LEN + 4 * NROOTS])
    frame = codeblock[:FRAME_LEN]
    par = codeblock[FRAME_LEN:]
    nerr, uncorrectable = 0, False
    for j in range(4):
        e = rs_nerrors(bytes(frame[j::4]) + bytes(par[j::4]))
        if e < 0:
            uncorrectable = True
        else:
            nerr += e
    return dict(asm=True, rs_errors=nerr, uncorrectable=uncorrectable,
                frame=frame)
