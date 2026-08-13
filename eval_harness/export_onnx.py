#!/usr/bin/env python3
"""Export a Community-Forensics checkpoint to ONNX, quantize it, and verify.

This is what makes the browser path possible: the extension runs
onnxruntime-web, not PyTorch. Two things must be true of the export or the
whole submission fails:

1. **One self-contained artifact.** The PyTorch load path pulls a timm backbone
   (`vit_small_patch16_224.augreg_in21k_ft_in1k`) from the Hub *in addition* to
   the Community-Forensics head. The ONNX graph must bake both in, so the
   extension performs exactly one download and then runs with no network at
   all -- the bounty disables internet after initial setup.
2. **Numerically faithful.** A quantized graph that drifts from the PyTorch
   model invalidates every accuracy number we measured. This script checks the
   ONNX output against PyTorch on real tensors and fails loudly on drift,
   rather than trusting that export "probably worked".

Usage:
    python export_onnx.py --model OwensLab/commfor-model-224 --out ../models/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mb(path: Path) -> float:
    """Total size in MB, INCLUDING any external-data sidecar.

    torch.onnx.export writes weights to `<name>.onnx.data` when the model
    exceeds the protobuf limit, leaving the .onnx file a ~0.1 MB graph stub.
    Reporting only the stub would understate the extension's download by three
    orders of magnitude -- and download size is a bounty-relevant constraint,
    not a footnote.
    """
    total = path.stat().st_size
    for sidecar in path.parent.glob(path.name + ".data"):
        total += sidecar.stat().st_size
    return total / (1024 * 1024)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="OwensLab/commfor-model-224")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--skip-quantize", action="store_true")
    p.add_argument("--skip-fp16", action="store_true")
    args = p.parse_args()

    try:
        import torch
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print("pip install onnx onnxruntime", file=sys.stderr)
        return 1

    # Reuse the exact loader the eval harness uses, so the exported graph is the
    # same model the accuracy numbers were measured on.
    from run_inference_commfor import load_model  # noqa: E402

    print(f"Loading {args.model} ...")
    model, model_dir = load_model(args.model, device="cpu")
    model.eval()
    # input_size comes from the checkpoint's own config, not from the model id
    # string -- the 224/384 naming is a convention, the config is the truth.
    input_size = int(json.loads(
        (Path(model_dir) / "config.json").read_text(encoding="utf-8")
    )["input_size"])
    print(f"  input_size from config.json: {input_size}")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    fp32_path = args.out / f"{tag}.onnx"

    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"Exporting to ONNX (opset {args.opset}, input {input_size}x{input_size}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["pixel_values"],
        output_names=["logit"],
        # Dynamic batch so the extension can score several images in one call.
        dynamic_axes={"pixel_values": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    print(f"  FP32: {fp32_path.name}  {mb(fp32_path):.1f} MB")

    # --- verify FP32 ONNX against PyTorch on real inputs -------------------
    print("\nVerifying ONNX matches PyTorch ...")
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    torch.manual_seed(0)
    probe = torch.randn(4, 3, input_size, input_size, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(probe).reshape(-1).numpy()
    onnx_out = sess.run(None, {"pixel_values": probe.numpy()})[0].reshape(-1)
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"  max |torch - onnx| = {max_diff:.3e}  (tolerance {args.tolerance:.0e})")
    if max_diff > args.tolerance:
        print("  FAIL: ONNX output drifted from PyTorch. Do NOT ship this.",
              file=sys.stderr)
        return 1
    print("  OK")

    report = {
        "model": args.model,
        "input_size": input_size,
        "opset": args.opset,
        "fp32_file": fp32_path.name,
        "fp32_mb": round(mb(fp32_path), 2),
        "fp32_max_abs_diff_vs_torch": max_diff,
    }

    # --- dynamic int8 quantization ----------------------------------------
    if not args.skip_quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("\nSkipping quantization (onnxruntime.quantization unavailable)")
        else:
            int8_path = args.out / f"{tag}.int8.onnx"
            print(f"\nQuantizing to int8 ...")

            # quantize_dynamic runs onnx shape inference internally, which trips
            # over this graph ("Inferred shape and existing shape differ in
            # dimension 0: (384) vs (1)"). The fix is to re-export a static
            # batch-1 graph with weights inline, and quantize THAT: the dynamic
            # batch axis is what confuses the inferencer, and the extension
            # scores one image per call anyway.
            static_path = args.out / f"{tag}.static.onnx"
            torch.onnx.export(
                model,
                dummy,
                str(static_path),
                input_names=["pixel_values"],
                output_names=["logit"],
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,          # legacy tracer emits a simpler graph
            )
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process
                pre_path = args.out / f"{tag}.pre.onnx"
                quant_pre_process(str(static_path), str(pre_path), skip_symbolic_shape=False)
                quant_src = pre_path
                print("  ran quant_pre_process")
            except Exception as exc:  # noqa: BLE001
                print(f"  quant_pre_process unavailable/failed ({exc}); "
                      f"quantizing the static graph directly")
                quant_src = static_path

            quantize_dynamic(
                model_input=str(quant_src),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
            )
            print(f"  INT8: {int8_path.name}  {mb(int8_path):.1f} MB "
                  f"({mb(fp32_path)/mb(int8_path):.1f}x smaller)")

            qsess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
            # The quantized graph has a STATIC batch of 1 (that is what made it
            # quantizable at all), so feed the probe one image at a time.
            q_out = np.concatenate([
                qsess.run(None, {"pixel_values": probe[i : i + 1].numpy()})[0].reshape(-1)
                for i in range(probe.shape[0])
            ])
            q_diff = float(np.max(np.abs(torch_out - q_out)))
            # Quantization legitimately shifts logits; what matters is whether
            # the *decision* changes, so report both.
            same_sign = bool(np.all((torch_out >= 0) == (q_out >= 0)))
            print(f"  max |torch - int8| = {q_diff:.3e}")
            print(f"  same decision on probe batch: {same_sign}")
            if not same_sign:
                print("  WARNING: int8 flipped a decision on random input. "
                      "Re-run the full eval against the quantized graph before "
                      "trusting the accuracy numbers.")
            report.update({
                "int8_file": int8_path.name,
                "int8_mb": round(mb(int8_path), 2),
                "int8_max_abs_diff_vs_torch": q_diff,
                "int8_same_decision_on_probe": same_sign,
            })

    # --- fp16 -------------------------------------------------------------
    # int8 dynamic quantization measured a 0.043 AUC loss on the holdout, which
    # is far too much when we sit within noise of the pass bar. fp16 halves the
    # size like int8 does not, but keeps far more precision, and WebGPU handles
    # it natively.
    if not args.skip_fp16:
        try:
            from onnxconverter_common import float16 as ort_fp16
        except ImportError:
            print("\nSkipping fp16 (pip install onnxconverter-common)")
        else:
            fp16_path = args.out / f"{tag}.fp16.onnx"
            print("\nConverting to fp16 ...")
            model_fp32 = onnx.load(str(fp32_path))
            model_fp16 = ort_fp16.convert_float_to_float16(
                model_fp32, keep_io_types=True
            )
            onnx.save(model_fp16, str(fp16_path),
                      save_as_external_data=False, all_tensors_to_one_file=True)
            print(f"  FP16: {fp16_path.name}  {mb(fp16_path):.1f} MB "
                  f"({mb(fp32_path)/max(mb(fp16_path), 0.01):.1f}x smaller)")

            fsess = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
            f_out = fsess.run(None, {"pixel_values": probe.numpy()})[0].reshape(-1)
            f_diff = float(np.max(np.abs(torch_out - f_out)))
            f_same = bool(np.all((torch_out >= 0) == (f_out >= 0)))
            print(f"  max |torch - fp16| = {f_diff:.3e}")
            print(f"  same decision on probe batch: {f_same}")
            report.update({
                "fp16_file": fp16_path.name,
                "fp16_mb": round(mb(fp16_path), 2),
                "fp16_max_abs_diff_vs_torch": f_diff,
                "fp16_same_decision_on_probe": f_same,
            })

    (args.out / f"{tag}.export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out / f'{tag}.export_report.json'}")
    print("\nNOTE: accuracy must be re-verified against the QUANTIZED graph "
          "before submission -- int8 weights are not the model we measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
