from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing_pipeline.pulse_ptbxl.pulse_inference import load_pulse_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PULSE-7B inference for PTB-XL 71-label ECG classification"
    )
    parser.add_argument("--image-file", required=True, help="Path to ECG image file")
    parser.add_argument(
        "--model-path",
        default="PULSE-ECG/PULSE-7B",
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device passed to the vendored PULSE loader",
    )
    parser.add_argument(
        "--conv-mode",
        default="llava_v1",
        help="Conversation template name for the vendored PULSE model",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Enable 4-bit loading when fp16 does not fit on the target GPU",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature; 0 keeps output near-deterministic",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for rationale plus labels",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--emit-asl-block",
        action="store_true",
        help="Append the stage-2 <asl_labels> block to the output",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the final prompt used for generation",
    )
    parser.add_argument(
        "--show-raw-output",
        action="store_true",
        help="Print the raw model output before parsing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_file = Path(args.image_file)
    classifier = load_pulse_classifier(
        model_path=args.model_path,
        device=args.device,
        conv_mode=args.conv_mode,
        load_4bit=args.load_4bit,
    )
    prediction = classifier.classify_image(
        image_file=image_file,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    if args.format == "json":
        payload = {
            "image_file": str(image_file),
            "labels": prediction.labels,
            "rationale": prediction.rationale,
            "rationale_source": prediction.rationale_source,
            "binary_vector": prediction.binary_vector,
            "unknown_tokens": prediction.unknown_tokens,
            "raw_output": prediction.raw_output,
        }
        if prediction.confidence_scores:
            payload["confidence_scores"] = prediction.confidence_scores
        if args.show_prompt:
            payload["prompt"] = prediction.prompt
        if args.emit_asl_block:
            payload["asl_block"] = prediction.as_asl_block()
        print(json.dumps(payload, indent=2))
        return

    labels_text = ", ".join(prediction.labels) if prediction.labels else "NONE"
    print(f"Image: {image_file}")
    print(f"Labels: {labels_text}")
    if prediction.confidence_scores:
        conf_text = ", ".join(f"{code} ({score:.2f})" for code, score in prediction.confidence_scores.items())
        print(f"Confidence: {conf_text}")
    print(f"Rationale: {prediction.rationale or 'N/A'}")
    print(f"Rationale source: {prediction.rationale_source}")
    if prediction.unknown_tokens:
        print(f"Unknown tokens: {', '.join(prediction.unknown_tokens)}")
    if args.show_prompt:
        print()
        print("Prompt:")
        print(prediction.prompt)
    if args.show_raw_output:
        print()
        print("Raw output:")
        print(prediction.raw_output or "<empty>")
    if args.emit_asl_block:
        print()
        print(prediction.as_asl_block())


if __name__ == "__main__":
    main()
