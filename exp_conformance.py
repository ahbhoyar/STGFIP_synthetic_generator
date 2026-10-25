"""Q1: encode telemetry to CADUs and round-trip through the decoder."""
import json, struct
import numpy as np
from stgfip.batch import CHANNELS, CH_SUB
from stgfip.ccsds import (space_packet, parse_space_packet, tm_frame,
                          parse_tm_frame, cadu, decadu, APIDS, FRAME_DATA,
                          crc16_ccitt)

RANGES = {}          # per-channel quantisation range, 16-bit


def quantise(v, lo, hi, bits=16):
    q = np.round((v - lo) * ((2 ** bits - 1) / (hi - lo)))
    return int(np.clip(q, 0, 2 ** bits - 1))


def dequantise(q, lo, hi, bits=16):
    return lo + q * (hi - lo) / (2 ** bits - 1)


def build_stream(F, t, n_samples, seed=0, ber=0.0, frame_loss=0.0):
    rng = np.random.default_rng(seed)
    for c in CHANNELS:
        lo, hi = float(F[c].min()), float(F[c].max())
        pad = 0.05 * (hi - lo) + 1e-6
        RANGES[c] = (lo - pad, hi + pad)
    by_sub = {}
    for c in CHANNELS:
        by_sub.setdefault(CH_SUB[c], []).append(c)

    cadus, truth, seqs = [], [], {s: 0 for s in by_sub}
    mc = vc = 0
    for i in range(n_samples):
        packets = []
        for sub, chans in by_sub.items():
            payload = b"".join(
                struct.pack(">H", quantise(F[c][i], *RANGES[c])) for c in chans)
            packets.append(space_packet(APIDS[sub], seqs[sub], payload, t[i]))
            seqs[sub] = (seqs[sub] + 1) & 0x3FFF
        blob = b"".join(packets)
        for off in range(0, len(blob), FRAME_DATA):
            fr = tm_frame(0x2A, 1, mc & 0xFF, vc & 0xFF, blob[off:off+FRAME_DATA])
            mc += 1; vc += 1
            cu = bytearray(cadu(fr))
            corrupted = 0
            if ber > 0:
                nbits = len(cu) * 8
                nflip = rng.binomial(nbits, ber)
                for _ in range(nflip):
                    p = rng.integers(4, len(cu))      # leave ASM intact
                    cu[p] ^= 1 << rng.integers(0, 8)
                corrupted = nflip
            if frame_loss > 0 and rng.random() < frame_loss:
                continue
            cadus.append(bytes(cu)); truth.append(corrupted)
    return cadus, truth, RANGES


def decode_all(cadus):
    asm_ok = rs_corr = rs_unc = crc_ok = 0
    seq_seen = {}
    gaps = 0
    values = []
    for cu in cadus:
        d = decadu(cu)
        if not d["asm"]:
            continue
        asm_ok += 1
        if d["uncorrectable"]:
            rs_unc += 1
        elif d["rs_errors"] > 0:
            rs_corr += 1
        fr = parse_tm_frame(d["frame"])
        if fr is None:
            continue
        if fr["crc_ok"]:
            crc_ok += 1
        buf = fr["data"]; off = 0
        while off < len(buf) - 6:
            p = parse_space_packet(buf[off:])
            if p is None or p["apid"] not in APIDS.values():
                break
            prev = seq_seen.get(p["apid"])
            if prev is not None and p["seq"] != (prev + 1) & 0x3FFF:
                gaps += 1
            seq_seen[p["apid"]] = p["seq"]
            values.append((p["apid"], p["body"][7:]))
            off += p["total"]
    return dict(asm=asm_ok, crc_ok=crc_ok, rs_corrected=rs_corr,
                rs_uncorrectable=rs_unc, seq_gaps=gaps, packets=len(values))


def main(mode, N=250):
    d = np.load("/tmp/stgfip_data_main/labelled.npz")
    t = d["t"]
    F = {c: d[f"F_{c}"][:, 0].astype(float) for c in CHANNELS}
    import os
    res = json.load(open("results_conformance.json")) \
        if os.path.exists("results_conformance.json") else {}
    if mode == "nominal":
        cad, _, rng_map = build_stream(F, t, N, seed=1, ber=0.0)
        r = decode_all(cad); r["cadus"] = len(cad)
        res["nominal"] = r; res["cadu_bytes"] = len(cad[0])
        errs = []
        for c in CHANNELS:
            lo, hi = rng_map[c]
            q = np.array([quantise(v, lo, hi) for v in F[c][:N]])
            back = dequantise(q, lo, hi)
            errs.append(float(np.max(np.abs(back - F[c][:N])) / (hi - lo)))
        res["roundtrip_rel_err"] = max(errs)
        res["quant_bound"] = 0.5 / (2 ** 16 - 1)
    else:
        cad, _, _ = build_stream(F, t, N, seed=2, ber=2e-5, frame_loss=0.004)
        r = decode_all(cad); r["cadus"] = len(cad)
        r["frames_lost"] = N - len(cad)
        res["faulted"] = r
    json.dump(res, open("results_conformance.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "nominal")
