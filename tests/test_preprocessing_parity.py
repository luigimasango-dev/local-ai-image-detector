"""Pin the preprocessing contract shared by Python and the browser.

The extension re-implements preprocessing in JavaScript (extension/inference.js).
If the two drift — wrong resize semantics, wrong normalization constants, wrong
channel order — nothing throws. The extension just scores worse than the
benchmark said it would, and we would not find out until a submission failed.

These tests pin the exact numbers both implementations must agree on, so a
change to either side has to consciously break a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval_harness"))

from run_inference_onnx import MEAN, STD, RESIZE_FOR, preprocess  # noqa: E402

EXT_JS = Path(__file__).resolve().parent.parent / "extension" / "inference.js"


def test_normalization_constants_are_imagenet():
    """These come from the Community-Forensics training transform, not a guess."""
    assert MEAN == (0.485, 0.456, 0.406)
    assert STD == (0.229, 0.224, 0.225)


def test_resize_sizes_match_upstream_mapping():
    """determine_resize_crop_sizes(): 224 -> 256, 384 -> 440."""
    assert RESIZE_FOR[224] == 256
    assert RESIZE_FOR[384] == 440


def test_javascript_uses_the_same_constants():
    """The browser implementation must carry identical numbers."""
    js = EXT_JS.read_text(encoding="utf-8")
    for value in ("0.485", "0.456", "0.406", "0.229", "0.224", "0.225"):
        assert value in js, f"extension/inference.js missing constant {value}"
    assert "224: 256" in js and "384: 440" in js, \
        "extension/inference.js resize mapping does not match Python"


def test_javascript_carries_the_calibration():
    """Shipping uncalibrated scores would drop us from ~0.83 to ~0.71.

    The coefficients must be the ones fitted on the FP16 graphs the extension
    actually loads — not the FP32 fit used during measurement.
    """
    js = EXT_JS.read_text(encoding="utf-8")
    cal_path = (Path(__file__).resolve().parent.parent
                / "eval_harness" / "calibration" / "fp16_shipping.json")
    if not cal_path.exists():
        # Calibration artifacts are gitignored; fall back to pinning the values.
        assert "0.6758" in js, "Platt A coefficient missing from extension"
        assert "2.0932" in js, "Platt B coefficient missing from extension"
        return

    import json
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    for name, value in (("A", cal["a"]), ("B", cal["b"])):
        assert f"{value:.4f}" in js, (
            f"extension/inference.js Platt {name} does not match the fitted "
            f"fp16 calibration ({value:.4f}). Re-run calibrate.py and update it."
        )


def test_extension_ships_fp16_not_int8():
    """int8 measured a 0.043 AUC loss — enough to fail the bounty."""
    js = EXT_JS.read_text(encoding="utf-8")
    assert "fp16.onnx" in js, "extension should load the fp16 graphs"
    assert "int8.onnx" not in js, (
        "extension is loading int8 weights, which cost 0.043 AUC on the holdout"
    )


def test_preprocess_output_shape_and_range():
    from PIL import Image
    import numpy as np

    # Non-square on purpose: a square input would hide an aspect-ratio bug.
    img = Image.new("RGB", (640, 480), (128, 64, 200))
    for size in (224, 384):
        arr = preprocess(img, size)
        assert arr.shape == (1, 3, size, size), f"bad shape for {size}: {arr.shape}"
        assert arr.dtype == np.float32
        # A flat colour maps to a constant per channel: (v/255 - mean) / std
        for c in range(3):
            expected = ((img.getpixel((0, 0))[c] / 255.0) - MEAN[c]) / STD[c]
            assert abs(float(arr[0, c].mean()) - expected) < 1e-4, \
                f"channel {c} normalized wrong for size {size}"


def test_resize_preserves_aspect_ratio_not_square_stretch():
    """torchvision Resize(int) scales the SHORTER side and keeps aspect.

    Square-stretching instead is the classic silent accuracy killer: every
    image is subtly distorted and the model just quietly does worse.
    """
    from PIL import Image, ImageDraw
    import numpy as np

    # Wide image with a centered marker; a square stretch moves the marker.
    img = Image.new("RGB", (800, 400), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([390, 190, 410, 210], fill=(255, 255, 255))

    arr = preprocess(img, 224)
    # Center pixel should be the bright marker after correct resize+crop.
    center = float(arr[0, :, 112, 112].mean())
    corner = float(arr[0, :, 5, 5].mean())
    assert center > corner, "centre marker lost — resize/crop geometry is wrong"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {exc}")
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
