#!/usr/bin/env python3
"""Export LSTM and YOLO models to ONNX format for edge deployment.

Exports to models/exports/ directory. Original PyTorch models are NOT modified.
Validates that ONNX outputs match PyTorch outputs within tolerance.

Usage:
    python scripts/export_onnx.py
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXPORTS_DIR = Path("models/exports")


def export_lstm(checkpoint_path: str = "models/checkpoints/best.pth") -> Path:
    """Export LSTM fall detector to ONNX."""
    from models.fall_detector import FallDetector

    logger.info("=" * 60)
    logger.info("Exporting LSTM to ONNX...")

    # Load model
    model = FallDetector()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Export
    dummy_input = torch.randn(1, 30, 5)
    onnx_path = EXPORTS_DIR / "fall_detector.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
    )

    size_kb = onnx_path.stat().st_size / 1024
    logger.info(f"  Exported: {onnx_path} ({size_kb:.0f} KB)")

    # Validate
    _validate_lstm(model, onnx_path)

    return onnx_path


def _validate_lstm(pytorch_model: torch.nn.Module, onnx_path: Path):
    """Validate ONNX LSTM output matches PyTorch."""
    import onnxruntime as rt

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    max_diff = 0.0
    for batch_size in [1, 4, 8]:
        test_input = np.random.randn(batch_size, 30, 5).astype(np.float32)

        # PyTorch
        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(test_input)).numpy()

        # ONNX
        onnx_out = sess.run(None, {input_name: test_input})[0]

        diff = np.abs(pt_out - onnx_out).max()
        max_diff = max(max_diff, diff)

        assert not np.any(np.isnan(onnx_out)), f"NaN in ONNX output (batch={batch_size})"
        assert not np.any(np.isinf(onnx_out)), f"Inf in ONNX output (batch={batch_size})"

    status = "PASS" if max_diff < 1e-4 else "FAIL"
    logger.info(f"  Validation: {status} (max diff: {max_diff:.2e})")

    if max_diff >= 1e-4:
        logger.error(f"  ONNX output differs from PyTorch by {max_diff:.2e}!")
        raise ValueError("ONNX validation failed")


def export_lstm_int8(fp32_path: Path) -> Path:
    """Create INT8 quantized version of the LSTM ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info("\nQuantizing LSTM to INT8...")

    int8_path = EXPORTS_DIR / "fall_detector_int8.onnx"

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    fp32_size = fp32_path.stat().st_size / 1024
    int8_size = int8_path.stat().st_size / 1024
    reduction = (1 - int8_size / fp32_size) * 100

    logger.info(f"  Exported: {int8_path} ({int8_size:.0f} KB)")
    logger.info(f"  Size reduction: {reduction:.0f}% ({fp32_size:.0f}KB -> {int8_size:.0f}KB)")

    # Validate INT8 accuracy
    _validate_int8(fp32_path, int8_path)

    return int8_path


def _validate_int8(fp32_path: Path, int8_path: Path):
    """Validate INT8 accuracy against FP32."""
    import onnxruntime as rt

    fp32_sess = rt.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_sess = rt.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = fp32_sess.get_inputs()[0].name

    # Test with 200 random samples
    np.random.seed(42)
    total = 200
    matches = 0

    for _ in range(total):
        test_input = np.random.randn(1, 30, 5).astype(np.float32)
        fp32_out = fp32_sess.run(None, {input_name: test_input})[0]
        int8_out = int8_sess.run(None, {input_name: test_input})[0]

        if np.argmax(fp32_out) == np.argmax(int8_out):
            matches += 1

    accuracy = matches / total * 100
    status = "PASS" if accuracy >= 98 else "FAIL"
    logger.info(f"  INT8 vs FP32 agreement: {accuracy:.1f}% ({status})")

    if accuracy < 98:
        logger.warning("  INT8 accuracy below 98% — consider using FP32 for production")


def export_yolo(model_name: str = "yolo11n-pose.pt") -> Path:
    """Export YOLO pose model to ONNX using Ultralytics built-in export."""
    from ultralytics import YOLO

    logger.info("\n" + "=" * 60)
    logger.info("Exporting YOLO to ONNX...")

    model = YOLO(model_name)

    export_path = model.export(
        format="onnx",
        simplify=True,
        opset=17,
        dynamic=True,
    )

    # Move to exports directory
    src = Path(export_path)
    dst = EXPORTS_DIR / src.name
    if src != dst:
        shutil.move(str(src), str(dst))

    size_mb = dst.stat().st_size / (1024 * 1024)
    logger.info(f"  Exported: {dst} ({size_mb:.1f} MB)")

    # Basic validation
    _validate_yolo(dst)

    return dst


def _validate_yolo(onnx_path: Path):
    """Basic validation that YOLO ONNX model loads and runs."""
    import onnxruntime as rt

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()

    logger.info(f"  Input: {input_info.name} shape={input_info.shape} type={input_info.type}")
    for out in output_info:
        logger.info(f"  Output: {out.name} shape={out.shape}")

    # Test inference with dummy image
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = sess.run(None, {input_info.name: test_input})

    assert outputs is not None, "YOLO ONNX returned None"
    assert len(outputs) > 0, "YOLO ONNX returned empty outputs"
    logger.info(f"  Validation: PASS (output shape: {outputs[0].shape})")


def main():
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("ONNX Model Export for Edge Deployment")
    logger.info("Models exported to: models/exports/ (originals untouched)")
    logger.info("")

    # 1. Export LSTM
    lstm_path = export_lstm()

    # 2. Export LSTM INT8 (optional — may fail on some ONNX graph structures)
    try:
        export_lstm_int8(lstm_path)
    except Exception as e:
        logger.warning(f"  INT8 quantization skipped: {e}")
        logger.info("  FP32 ONNX model is still valid for edge deployment")

    # 3. Export YOLO
    export_yolo()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 60)

    for f in sorted(EXPORTS_DIR.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            logger.info(f"  {f.name:<35} {size / (1024*1024):.1f} MB")
        else:
            logger.info(f"  {f.name:<35} {size / 1024:.0f} KB")

    total_mb = sum(f.stat().st_size for f in EXPORTS_DIR.iterdir()) / (1024 * 1024)
    logger.info(f"\n  Total edge model size: {total_mb:.1f} MB")
    logger.info("  Original PyTorch models: UNTOUCHED")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
