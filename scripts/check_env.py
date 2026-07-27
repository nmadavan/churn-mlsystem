"""Check that required packages are importable; install only the missing ones.

Run:
    python scripts/check_env.py            # report only
    python scripts/check_env.py --install  # pip install whatever is missing
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# requirements.txt uses distribution names; a few differ from their import name.
IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "uvicorn": "uvicorn",
}

REQ_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


def parse_requirements(path):
    """Return list of (dist_name, pinned_spec) from requirements.txt, skipping comments."""
    reqs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dist = line.split("==")[0].split(">=")[0].strip()
        reqs.append((dist, line))
    return reqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true",
                        help="pip install packages that fail to import")
    args = parser.parse_args()

    reqs = parse_requirements(REQ_FILE)
    missing = []
    for dist, spec in reqs:
        module = IMPORT_NAME.get(dist, dist)
        try:
            importlib.import_module(module)
            print(f"[ok]      {dist}")
        except ImportError:
            print(f"[missing] {dist}")
            missing.append(spec)

    if not missing:
        print("\nAll dependencies present.")
        return 0

    if not args.install:
        print(f"\n{len(missing)} missing. Re-run with --install to install them.")
        return 1

    print(f"\nInstalling {len(missing)} missing package(s)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
