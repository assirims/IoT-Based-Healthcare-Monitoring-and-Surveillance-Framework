"""Tiny reader for PhysioNet MIT format files (header .hea, signal format 212, annotation .atr) - no wfdb package needed."""
import numpy as np

# WFDB annotation codes used by the MIT-BIH Arrhythmia Database
CODE2SYM = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E", 11: "j", 12: "/",
            13: "Q", 14: "~", 16: "|", 18: "s", 19: "T", 22: '"', 28: "+", 34: "e", 37: "x", 38: "f", 39: "[",
            40: "]", 31: "!", 32: "[", 33: "]"}
# AAMI EC57 beat classes
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V",
        "F": "F",
        "/": "Q", "f": "Q", "Q": "Q"}


def read_header(txt):
    lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
    rec = lines[0].split()
    nsig, fs, nsamp = int(rec[1]), float(rec[2]), int(rec[3])
    sigs = []
    for l in lines[1:1 + nsig]:
        p = l.split()
        gain = float(p[2].split("(")[0].split("/")[0]) if len(p) > 2 else 200.0
        baseline = int(p[2].split("(")[1].split(")")[0]) if len(p) > 2 and "(" in p[2] else (int(p[4]) if len(p) > 4 else 0)
        sigs.append(dict(file=p[0], fmt=p[1], gain=gain or 200.0, adczero=int(p[4]) if len(p) > 4 else 0,
                         baseline=baseline, desc=" ".join(p[8:]) if len(p) > 8 else ""))
    return dict(nsig=nsig, fs=fs, nsamp=nsamp, signals=sigs)


def read_212(raw, nsig=2):
    b = np.frombuffer(raw, dtype=np.uint8)
    b = b[: (len(b) // 3) * 3].reshape(-1, 3).astype(np.int16)
    s1 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s2 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    s1 = np.where(s1 > 2047, s1 - 4096, s1)
    s2 = np.where(s2 > 2047, s2 - 4096, s2)
    x = np.empty(len(s1) * 2, np.int16)
    x[0::2], x[1::2] = s1, s2
    return x.reshape(-1, nsig)


def read_atr(raw):
    """Returns sample indices, symbols and aux strings (rhythm labels)."""
    w = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<u2")
    t, i = 0, 0
    samples, syms, aux = [], [], []
    while i < len(w):
        a, v = int(w[i]) >> 10, int(w[i]) & 0x3FF
        i += 1
        if a == 0 and v == 0:
            break
        if a == 59:                                   # SKIP: 32-bit interval follows (high word first)
            t += (int(w[i]) << 16) | int(w[i + 1])
            i += 2
            continue
        if a == 63:                                   # AUX string attached to the previous annotation
            nb = v
            s = raw[2 * i: 2 * i + nb].decode("latin1").rstrip("\x00")
            if aux:
                aux[-1] = s
            i += (nb + 1) // 2
            continue
        if a in (60, 61, 62):                         # NUM / SUB / CHN modifiers
            continue
        t += v
        samples.append(t)
        syms.append(CODE2SYM.get(a, "?"))
        aux.append("")
    return np.array(samples), np.array(syms), np.array(aux)
