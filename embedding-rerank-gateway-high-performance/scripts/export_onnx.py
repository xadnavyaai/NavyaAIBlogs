#!/usr/bin/env python3
"""
Export sentence-transformers/all-MiniLM-L6-v2 to ONNX for use by the Rust gateway.
Requires: pip install optimum[onnx] transformers
Output: model.onnx + tokenizer files in the given output dir (e.g. ../rust-gateway/model).
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export all-MiniLM-L6-v2 to ONNX")
    parser.add_argument("--output", "-o", type=Path, default=Path(__file__).resolve().parent.parent / "rust-gateway" / "model",
                        help="Output directory for model.onnx and tokenizer files")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Model ID or path")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("Install optimum and transformers: pip install 'optimum[onnx]' transformers", file=sys.stderr)
        sys.exit(1)

    # Feature extraction (embeddings). See HF discussion on all-MiniLM-L6-v2 ONNX export.
    print(f"Exporting {args.model} to {out} ...")
    try:
        main_export(
            args.model,
            output=str(out),
            task="feature-extraction",
            opset=14,
        )
    except Exception as e:
        print(f"Export with task=feature-extraction failed: {e}", file=sys.stderr)
        print("Trying default task...", file=sys.stderr)
        main_export(args.model, output=str(out), opset=14)
    print("Done. Files in", out)
    for f in sorted(out.iterdir()):
        print(" ", f.name)

if __name__ == "__main__":
    main()
