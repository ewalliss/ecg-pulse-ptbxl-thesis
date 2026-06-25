"""
Interactive PULSE-7B inference REPL.

Usage
-----
    python -m src.run_pulse_chat
    python -m src.run_pulse_chat --model-path model/models--PULSE-ECG--PULSE-7B --load-4bit

Input syntax (inside the REPL)
-------------------------------
    @/path/to/ecg.png          attach an image (required before running)
    <any text>                 custom question/prompt (optional — uses default if blank)
    Enter on blank line        run inference with current image + prompt
    new                        clear image and prompt, start fresh
    quit / exit / q            exit

Examples
--------
    > @/tmp/00001.png
    > What SCP codes are present?
    > (press Enter on blank line to run)

    > @outputs/ecg_images/00042.png
    > (press Enter immediately to use default classification prompt)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_AT_PATTERN = re.compile(r"@(\S+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive PULSE-7B inference REPL")
    p.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conv-mode", default="llava_v1")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def _print_result(pred) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    if pred.rationale:
        print(f"Reasoning:\n  {pred.rationale}\n")
    labels = ", ".join(pred.labels) if pred.labels else "NONE"
    print(f"Labels:  {labels}")
    if pred.confidence_scores:
        top = sorted(pred.confidence_scores.items(), key=lambda x: -x[1])[:8]
        conf_str = "  ".join(f"{c}({s:.2f})" for c, s in top)
        print(f"Conf:    {conf_str}")
    if pred.unknown_tokens:
        print(f"Unknown: {', '.join(pred.unknown_tokens)}")
    print(sep)


def _read_block(prompt_prefix: str) -> tuple[Path | None, str]:
    """Read lines until blank line. Returns (image_path, custom_prompt)."""
    image_path: Path | None = None
    lines: list[str] = []

    print(f"\n{prompt_prefix}")
    print("  Use @/path/to/image.png to attach image.")
    print("  Press Enter on a blank line to run.  'new' to reset.  'quit' to exit.\n")

    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None, "quit"

        if line.lower() in ("quit", "exit", "q"):
            return None, "quit"
        if line.lower() == "new":
            return None, "new"

        # Extract @path anywhere in the line
        match = _AT_PATTERN.search(line)
        if match:
            candidate = Path(match.group(1))
            if candidate.exists():
                image_path = candidate
                print(f"  [image set] {image_path}")
            else:
                print(f"  [!] File not found: {candidate}")
            # Strip the @ref from the remaining text
            rest = _AT_PATTERN.sub("", line).strip()
            if rest:
                lines.append(rest)
            continue

        if line == "":
            # Blank line = run
            break

        lines.append(line)

    custom_prompt = " ".join(lines).strip()
    return image_path, custom_prompt


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  PULSE-7B Interactive ECG Inference")
    print("=" * 60)
    print(f"  Model : {args.model_path}")
    print(f"  Device: {args.device}  4-bit: {args.load_4bit}")
    print("  Loading model — this may take 1-2 min...")

    from preprocessing_pipeline.pulse_ptbxl.pulse_inference import load_pulse_classifier

    classifier = load_pulse_classifier(
        model_path=args.model_path,
        device=args.device,
        conv_mode=args.conv_mode,
        load_4bit=args.load_4bit,
    )
    print("  Model ready.\n")

    current_image: Path | None = None
    session = 0

    while True:
        session += 1
        prefix = f"[{session}] New query"
        if current_image:
            prefix += f"  (current image: {current_image.name})"

        image, prompt = _read_block(prefix)

        if prompt == "quit":
            print("\nBye.")
            sys.exit(0)

        if prompt == "new":
            current_image = None
            print("  [reset] Image and prompt cleared.\n")
            continue

        # Allow reusing the last image if none given this round
        if image is not None:
            current_image = image

        if current_image is None:
            print("  [!] No image set. Use @/path/to/image.png first.\n")
            continue

        print(f"\n  Running inference on: {current_image}")
        print("  (generating...)")

        pred = classifier.classify_image(
            image_file=current_image,
            custom_prompt=prompt if prompt else None,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        _print_result(pred)


if __name__ == "__main__":
    main()
