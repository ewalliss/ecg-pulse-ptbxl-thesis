from __future__ import annotations

import json
from pathlib import Path

from preprocessing_pipeline.run_pulse_ptbxl_inference import main
from preprocessing_pipeline.pulse_ptbxl.pulse_inference import PulsePrediction


class _FakeClassifier:
    def __init__(self, prediction: PulsePrediction) -> None:
        self.prediction = prediction
        self.calls: list[dict[str, object]] = []

    def classify_image(
        self,
        image_file: str | Path,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
    ) -> PulsePrediction:
        self.calls.append(
            {
                "image_file": Path(image_file),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.prediction


def test_cli_json_output(monkeypatch, capsys, tmp_path):
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"not-a-real-image")

    fake_prediction = PulsePrediction(
        labels=["NORM", "SR"],
        rationale="Short explanation.",
        binary_vector=[1, 1] + [0] * 69,
        unknown_tokens=["UNKNOWNCODE"],
        raw_output="Rationale: Short explanation.\nLabels: NORM, SR",
        prompt="Prompt text",
        rationale_source="exact",
    )
    fake_classifier = _FakeClassifier(fake_prediction)

    monkeypatch.setattr(
        "src.run_pulse_ptbxl_inference.load_pulse_classifier",
        lambda **_: fake_classifier,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pulse_ptbxl_inference",
            "--image-file",
            str(image_file),
            "--format",
            "json",
            "--emit-asl-block",
            "--temperature",
            "0.1",
            "--max-new-tokens",
            "64",
        ],
    )

    main()
    output = json.loads(capsys.readouterr().out)

    assert output["image_file"] == str(image_file)
    assert output["labels"] == ["NORM", "SR"]
    assert output["rationale"] == "Short explanation."
    assert output["rationale_source"] == "exact"
    assert output["unknown_tokens"] == ["UNKNOWNCODE"]
    assert "asl_block" in output
    assert fake_classifier.calls == [
        {
            "image_file": image_file,
            "temperature": 0.1,
            "max_new_tokens": 64,
        }
    ]


def test_cli_text_output(monkeypatch, capsys, tmp_path):
    image_file = tmp_path / "sample.png"
    image_file.write_bytes(b"not-a-real-image")

    fake_prediction = PulsePrediction(
        labels=[],
        rationale="No supported code.",
        binary_vector=[0] * 71,
        unknown_tokens=[],
        raw_output="Rationale: No supported code.\nLabels: NONE",
    )
    fake_classifier = _FakeClassifier(fake_prediction)

    monkeypatch.setattr(
        "src.run_pulse_ptbxl_inference.load_pulse_classifier",
        lambda **_: fake_classifier,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pulse_ptbxl_inference",
            "--image-file",
            str(image_file),
        ],
    )

    main()
    output = capsys.readouterr().out

    assert f"Image: {image_file}" in output
    assert "Labels: NONE" in output
    assert "Rationale: No supported code." in output
