#!/usr/bin/env python3
"""
Sync the static frontend (``app/static``) into ``docs/`` for GitHub Pages.

``docs/`` must be a byte-identical copy of ``app/static``.  Run

    python scripts/build_pages.py          # sync (copy changed, delete stale)
    python scripts/build_pages.py --check  # exit 1 if docs/ is stale (used in CI)

Server-only paths are excluded from the copy.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "app", "static")
DST = os.path.join(ROOT, "docs")
EXCLUDE_DIRS = {"__pycache__"}
EXCLUDE_FILES = {".DS_Store"}


def _walk(base: str):
    out = set()
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_FILES:
                continue
            out.add(os.path.relpath(os.path.join(dirpath, fn), base))
    return out


def diff():
    src, dst = _walk(SRC), _walk(DST) if os.path.isdir(DST) else set()
    missing = sorted(src - dst)
    stale = sorted(dst - src)
    changed = sorted(p for p in src & dst
                     if not filecmp.cmp(os.path.join(SRC, p), os.path.join(DST, p), shallow=False))
    return missing, changed, stale


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report; exit 1 when docs/ differs from app/static")
    args = ap.parse_args(argv)
    missing, changed, stale = diff()
    if args.check:
        for label, items in (("missing", missing), ("changed", changed), ("stale", stale)):
            for p in items:
                print(f"{label}: docs/{p}")
        if missing or changed or stale:
            print("docs/ is out of date; run: python scripts/build_pages.py", file=sys.stderr)
            return 1
        print("docs/ is in sync with app/static")
        return 0
    for p in missing + changed:
        os.makedirs(os.path.dirname(os.path.join(DST, p)), exist_ok=True)
        shutil.copy2(os.path.join(SRC, p), os.path.join(DST, p))
    for p in stale:
        os.remove(os.path.join(DST, p))
    for dirpath, dirnames, filenames in sorted(os.walk(DST), key=lambda w: -len(w[0])):
        if dirpath != DST and not os.listdir(dirpath):
            os.rmdir(dirpath)
    print(f"synced docs/: {len(missing)} added, {len(changed)} updated, {len(stale)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
