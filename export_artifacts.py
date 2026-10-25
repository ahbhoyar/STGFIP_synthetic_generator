#!/usr/bin/env python3
"""
Export the small, human-readable artifacts the paper promises to release:
fixed splits, the channel/noise table, the fault schedule, the seeds, and a
real CADU byte stream. Writes to ./artifacts/ (a few hundred kB total).

Run after stage 2 of run_all.py:

    python3 export_artifacts.py
"""
import csv
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts")
DATA = "/tmp/stgfip_data_main"

sys.path.insert(0, HERE)
from stgfip.batch import CHANNELS                      # noqa: E402
from stgfip import ccsds                               # noqa: E402
from features import episode_split                     # noqa: E402
import gen_corpus                                      # noqa: E402


def need(path, hint):
    if not os.path.exists(path):
        sys.exit("missing %s\n  %s" % (path, hint))


def main():
    need(DATA + "/episodes.pkl",
         "run: python3 run_all.py --stage 1 && python3 run_all.py --stage 2")
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. fixed splits (episode indices, not rows) -----------------------
    tr, va, te = episode_split(96, seed=3)
    json.dump({"n_episodes": 96, "split": "60/15/25 by episode", "seed": 3,
               "train": sorted(int(i) for i in tr),
               "val":   sorted(int(i) for i in va),
               "test":  sorted(int(i) for i in te)},
              open(os.path.join(OUT, "splits.json"), "w"), indent=1)

    # ---- 2. channel list with the specified noise constants ---------------
    with open(os.path.join(OUT, "channels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "channel", "subsystem", "sigma_0"])
        for i, c in enumerate(CHANNELS):
            w.writerow([i, c, c.split("_")[0].upper(),
                        gen_corpus.NOISE[c]])

    # ---- 3. fault schedule, one row per episode ---------------------------
    eps = pickle.load(open(DATA + "/episodes.pkl", "rb"))
    keys = ["kind", "subsystem", "channel", "param", "ftype",
            "t_f", "severity"]
    with open(os.path.join(OUT, "fault_schedule.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + keys)
        for i, e in enumerate(eps):
            w.writerow([i] + [e.get(k, "") for k in keys])

    # ---- 4. seeds ----------------------------------------------------------
    json.dump({"corpus_seed": gen_corpus.SEED,
               "split_seed": 3,
               "bootstrap_seed": 0,
               "bootstrap_resamples": 400,
               "model_random_state": 0,
               "note": "RK4 h=1 s; detector decides once per 12 s"},
              open(os.path.join(OUT, "seeds.json"), "w"), indent=1)

    # ---- 5. a real CADU stream (the Layer-3 output the paper describes) ----
    rng = np.random.default_rng(0)
    lab = np.load(DATA + "/labelled.npz")
    n_cadu, seq, stream = 250, 0, bytearray()
    for i in range(n_cadu):
        payload = bytearray()
        # 8 packets per frame, one APID per subsystem, 16-bit quantised
        for apid in range(8):
            vals = (lab["y"][i * 8 + apid, :11] if "y" in lab
                    else rng.random(11))
            q = np.clip((np.asarray(vals, dtype=float) * 1000 + 32768),
                        0, 65535).astype(">u2").tobytes()
            payload += ccsds.space_packet(0x100 + apid, seq & 0x3FFF, q,
                                          obt=i)
            seq += 1
        payload = payload[:ccsds.FRAME_DATA].ljust(ccsds.FRAME_DATA, b"\x00")
        stream += ccsds.cadu(ccsds.tm_frame(0x5A, 1, i & 0xFF, i & 0xFF,
                                            bytes(payload)))
    open(os.path.join(OUT, "sample_250.cadu"), "wb").write(bytes(stream))

    print("wrote %s/" % OUT)
    for f in sorted(os.listdir(OUT)):
        print("  %-22s %8d B" % (f, os.path.getsize(os.path.join(OUT, f))))
    assert len(stream) == n_cadu * 1024, "CADU length is not 1024 B"
    print("\n%d CADUs x 1024 B = %d B (matches Sec. III-E)"
          % (n_cadu, len(stream)))


if __name__ == "__main__":
    main()
