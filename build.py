#!/usr/bin/env python3
"""Build the extension from source — one command, clean clone to working add-on.

    python build.py

The bounty requires the submission to be reproducible from source and states
that maintainers will independently build it. A fresh clone contains no model
weights (they are large and gitignored), so this script fetches the published
checkpoints, converts them to the fp16 ONNX graphs the extension runs, and
vendors the onnxruntime-web runtime into extension/lib/.

Everything downloaded here comes from public, permissively-licensed sources:

  * OwensLab/commfor-model-224 and -384 — MIT (Community Forensics, CVPR 2025)
  * onnxruntime-web — MIT (Microsoft)

After this completes, load `extension/` via chrome://extensions -> Load
unpacked. The extension then makes NO network requests at all.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Windows consoles default to a legacy code page (cp1252), and any non-ASCII
# character in output crashes the build with UnicodeEncodeError. Force UTF-8 so
# the script behaves the same on every grader's machine, and keep printed
# strings ASCII-only as a second line of defence.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "extension"
MODELS_OUT = EXT / "models"
LIB_OUT = EXT / "lib"

ORT_VERSION = "1.20.1"
ORT_TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"

# Only the files the extension actually loads. The full dist is ~90MB; these
# are the WebGPU-capable bundle plus its WASM binaries.
ORT_FILES = [
    "ort.webgpu.min.js",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
]

CHECKPOINTS = [
    ("OwensLab/commfor-model-224", "commfor-model-224"),
    ("OwensLab/commfor-model-384", "commfor-model-384"),
]

# The checkpoints load only through this repo's own model class.
COMMFOR_REPO = "https://github.com/JeongsooP/Community-Forensics.git"
CLONE_DIR = ROOT / "_commfor_src"

# Exact versions are pinned in requirements-build.txt — see the note there on
# why. This list is only used to produce a helpful error message when something
# is missing.
PY_DEPS_HINT = "pip install -r requirements-build.txt"


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def check_deps() -> bool:
    missing = []
    for mod, pip_name in [("torch", "torch"), ("timm", "timm"), ("onnx", "onnx"),
                          ("onnxruntime", "onnxruntime"), ("onnxscript", "onnxscript"),
                          ("onnxconverter_common", "onnxconverter-common"),
                          ("huggingface_hub", "huggingface_hub"), ("PIL", "pillow"),
                          ("numpy", "numpy")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("Missing Python packages:", " ".join(missing))
        print("\nInstall the exact tested versions with:")
        print(f"  {PY_DEPS_HINT}")
        print("\n(torch is CPU-only; no GPU is needed to build or to run this:")
        print("  pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu )")
        return False
    return True


def fetch_ort() -> None:
    step(f"Vendoring onnxruntime-web {ORT_VERSION} into extension/lib/")
    LIB_OUT.mkdir(parents=True, exist_ok=True)

    if all((LIB_OUT / f).exists() for f in ORT_FILES):
        print("  already present, skipping")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "ort.tgz"
        print(f"  downloading {ORT_TARBALL}")
        urllib.request.urlretrieve(ORT_TARBALL, tgz)
        with tarfile.open(tgz) as tar:
            tar.extractall(tmp)
        dist = Path(tmp) / "package" / "dist"
        for name in ORT_FILES:
            src = dist / name
            if not src.exists():
                raise SystemExit(f"  MISSING {name} in the npm package — "
                                 f"did onnxruntime-web change its layout?")
            shutil.copy(src, LIB_OUT / name)
            print(f"  {name}  ({(LIB_OUT / name).stat().st_size / 1e6:.1f} MB)")


def fetch_commfor_source() -> None:
    """Clone the Community-Forensics repo.

    Its published checkpoints are saved with PyTorchModelHubMixin and can only
    be loaded through the repo's own `ViTClassifier` class, so the source is a
    genuine build dependency, not just a reference. A clean clone does not have
    it (it is gitignored), which is precisely the gap that makes an untested
    build script fail on a grader's machine.
    """
    step("Fetching Community-Forensics source (needed to load the checkpoints)")
    if CLONE_DIR.exists() and (CLONE_DIR / "models.py").exists():
        print("  already present, skipping")
        return

    if shutil.which("git") is None:
        raise SystemExit(
            "  git not found on PATH. Install git, or manually clone\n"
            f"  {COMMFOR_REPO}\n  into {CLONE_DIR}"
        )

    print(f"  git clone {COMMFOR_REPO}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", COMMFOR_REPO, str(CLONE_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"  clone failed:\n{result.stderr.strip()}")

    if not (CLONE_DIR / "models.py").exists():
        raise SystemExit(
            f"  cloned, but {CLONE_DIR / 'models.py'} is missing — "
            "the upstream repo layout may have changed."
        )
    print(f"  cloned into {CLONE_DIR.name}/")


def build_models(skip_existing: bool) -> None:
    step("Exporting model checkpoints to fp16 ONNX")
    MODELS_OUT.mkdir(parents=True, exist_ok=True)

    for repo, tag in CHECKPOINTS:
        target = MODELS_OUT / f"{tag}.fp16.onnx"
        if skip_existing and target.exists():
            print(f"  {target.name} already present, skipping")
            continue

        print(f"  building {repo} …")
        result = subprocess.run(
            [sys.executable, str(ROOT / "eval_harness" / "export_onnx.py"),
             "--model", repo, "--skip-quantize", "--out", str(ROOT / "models")],
            cwd=str(ROOT),
            env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            raise SystemExit(f"  export failed for {repo}")

        src = ROOT / "models" / f"{tag}.fp16.onnx"
        if not src.exists():
            raise SystemExit(f"  expected {src} was not produced")
        shutil.copy(src, target)
        print(f"  {target.name}  ({target.stat().st_size / 1e6:.1f} MB)")


def verify() -> bool:
    step("Verifying the built extension")
    ok = True

    required = [
        EXT / "manifest.json", EXT / "background.js", EXT / "offscreen.html",
        EXT / "offscreen.js", EXT / "inference.js", EXT / "capabilities.js",
        EXT / "content.js", EXT / "popup.html", EXT / "popup.js", EXT / "styles.css",
    ] + [LIB_OUT / f for f in ORT_FILES] + [
        MODELS_OUT / f"{tag}.fp16.onnx" for _, tag in CHECKPOINTS
    ]

    for path in required:
        if not path.exists():
            print(f"  MISSING {path.relative_to(ROOT)}")
            ok = False

    if ok:
        try:
            json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  manifest.json is not valid JSON: {exc}")
            ok = False

    # The model filenames referenced in code must be the ones on disk.
    js = (EXT / "inference.js").read_text(encoding="utf-8")
    for _, tag in CHECKPOINTS:
        if f"{tag}.fp16.onnx" not in js:
            print(f"  inference.js does not reference {tag}.fp16.onnx")
            ok = False

    if ok:
        total = sum(p.stat().st_size for p in EXT.rglob("*") if p.is_file())
        print(f"  all files present — extension/ is {total / 1e6:.0f} MB")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="Rebuild models even if they already exist")
    p.add_argument("--skip-deps-check", action="store_true")
    args = p.parse_args()

    print(__doc__)

    if not args.skip_deps_check and not check_deps():
        return 1

    fetch_ort()
    fetch_commfor_source()
    build_models(skip_existing=not args.force)

    if not verify():
        print("\nBUILD FAILED — see missing items above.")
        return 1

    print("\n" + "=" * 62)
    print("BUILD OK")
    print("=" * 62)
    print("\nNext: open chrome://extensions, turn on Developer mode,")
    print(f"click 'Load unpacked' and select:\n  {EXT}")
    print("\nSee INSTALL.md for step-by-step instructions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
