
import hashlib
import os
import sys
import urllib.request

from config import DATA_PATH, RAW, URLS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(key):
    dst = os.path.join(DATA_PATH, RAW[key])
    if os.path.exists(dst):
        print("present:", dst)
        return dst
    print("downloading", URLS[key], "->", dst, flush=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(URLS[key]) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dst)
    return dst


def reassemble():
    """Large archives are shipped split into <25 MB parts (name.zip.part01, part02, ...): join them back."""
    import glob
    for key in ["fog", "har70", "ecg", "cog"]:
        dst = os.path.join(DATA_PATH, RAW[key])
        parts = sorted(glob.glob(dst + ".part*"))
        if parts and not os.path.exists(dst):
            with open(dst, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        out.write(f.read())
            print("re-assembled", dst, "from", len(parts), "parts")


def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    reassemble()
    keys = ["fog", "har70", "ecg", "cog"]
    have_falls = os.path.exists(os.path.join(DATA_PATH, RAW["falls"])) or os.path.isdir(os.path.join(DATA_PATH, "falls_selected"))
    if not have_falls or "--all" in sys.argv:
        keys.append("falls_zip")
    for k in keys:
        fetch(k)
    with open(os.path.join(DATA_PATH, "SHA256SUMS.txt"), "w") as f:
        for n in sorted(os.listdir(DATA_PATH)):
            p = os.path.join(DATA_PATH, n)
            if os.path.isfile(p) and n != "SHA256SUMS.txt":
                f.write(f"{sha256(p)}  {n}\n")
    print("checksums written")


if __name__ == "__main__":
    main()
