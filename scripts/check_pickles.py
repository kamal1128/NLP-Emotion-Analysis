"""scripts/check_pickles.py
Utility to inspect and (optionally) fix pickle files in models/.

Usage:
    python scripts/check_pickles.py      # just inspect
    python scripts/check_pickles.py --fix  # strip UTF-8 BOM if present and retest
"""
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FILES = [MODELS / "emotion_model.pkl", MODELS / "vectorizer.pkl"]

def inspect_and_test(path: Path):
    print(path.name)
    if not path.exists():
        print("  MISSING")
        return False
    b = path.read_bytes()
    head = b[:8]
    print("  size=", len(b))
    print("  head_hex=", head.hex())
    print("  starts_with_pickle_protocol=", head.startswith(b"\x80"))
    print("  starts_with_utf8_bom=", head.startswith(b"\xef\xbb\xbf"))
    # try to unpickle safely
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        print("  unpickle: OK (type=", type(obj), ")")
        return True
    except Exception as e:
        print("  unpickle: FAILED ->", repr(e))
        return False

def strip_bom(path: Path):
    b = path.read_bytes()
    if b.startswith(b"\xef\xbb\xbf"):
        print("  Stripping BOM from", path.name)
        path.write_bytes(b[3:])
        return True
    return False

def main():
    fix = "--fix" in sys.argv
    all_ok = True
    for p in FILES:
        ok = inspect_and_test(p)
        all_ok = all_ok and ok
        if not ok and fix:
            if strip_bom(p):
                print("  Retesting after strip:")
                ok2 = inspect_and_test(p)
                all_ok = all_ok and ok2
    return 0 if all_ok else 2

if __name__ == '__main__':
    sys.exit(main())
