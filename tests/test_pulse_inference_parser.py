from preprocessing_pipeline.config.preprocessing_config import NUM_CLASSES, SCP_ALL_CODES
from preprocessing_pipeline.pulse_ptbxl.pulse_inference import (
    PulsePrediction,
    build_classification_prompt,
    parse_classification_output,
)


def test_build_prompt_contains_all_canonical_codes():
    prompt = build_classification_prompt()

    assert "Rationale:" in prompt
    assert "Labels:" in prompt
    assert SCP_ALL_CODES[0] in prompt
    assert SCP_ALL_CODES[-1] in prompt


def test_parse_classification_output_extracts_labels_and_rationale():
    prediction = parse_classification_output(
        "Rationale: Sinus rhythm with inferior infarction pattern.\n"
        "Labels: IMI, NORM, SR"
    )

    assert prediction.rationale == "Sinus rhythm with inferior infarction pattern."
    assert prediction.rationale_source == "exact"
    assert prediction.labels == [code for code in SCP_ALL_CODES if code in {"NORM", "SR", "IMI"}]
    assert len(prediction.binary_vector) == NUM_CLASSES
    assert sum(prediction.binary_vector) == 3
    assert prediction.unknown_tokens == []


def test_parse_classification_output_tracks_unknown_tokens():
    prediction = parse_classification_output(
        "Rationale: Example.\n"
        "Labels: NORM, fake_label, SR, made-up"
    )

    assert prediction.labels == [code for code in SCP_ALL_CODES if code in {"NORM", "SR"}]
    assert prediction.unknown_tokens == ["FAKELABEL", "MADEUP"]


def test_parse_classification_output_accepts_reasoning_header():
    prediction = parse_classification_output(
        "Reasoning: ST elevation is visible in the inferior leads.\n"
        "Labels: IMI"
    )

    assert prediction.rationale == "ST elevation is visible in the inferior leads."
    assert prediction.rationale_source == "exact"
    assert prediction.labels == [code for code in SCP_ALL_CODES if code in {"IMI"}]


def test_parse_classification_output_uses_prose_before_labels_as_fallback():
    prediction = parse_classification_output(
        "Inferior Q waves and sinus rhythm support the chosen labels.\n"
        "Labels: IMI, SR"
    )

    assert prediction.rationale == "Inferior Q waves and sinus rhythm support the chosen labels."
    assert prediction.rationale_source == "fallback"
    assert prediction.labels == [code for code in SCP_ALL_CODES if code in {"IMI", "SR"}]


def test_parse_classification_output_handles_none_label():
    prediction = parse_classification_output(
        "Rationale: No supported SCP code is clearly present.\n"
        "Labels: NONE"
    )

    assert prediction.labels == []
    assert len(prediction.binary_vector) == NUM_CLASSES
    assert sum(prediction.binary_vector) == 0


def test_parse_classification_output_handles_i_dont_know_label():
    prediction = parse_classification_output(
        "Rationale: The image is not clear enough for a supported diagnosis.\n"
        "Labels: I DON'T KNOW"
    )

    assert prediction.labels == []
    assert prediction.unknown_tokens == []
    assert prediction.rationale_source == "exact"


def test_parse_classification_output_scans_raw_text_when_labels_line_missing():
    prediction = parse_classification_output(
        "Rationale: Consider NORM with SR and old IMI changes."
    )

    assert prediction.labels == [code for code in SCP_ALL_CODES if code in {"NORM", "SR", "IMI"}]
    assert prediction.rationale == "Consider NORM with SR and old IMI changes."


def test_parse_classification_output_marks_missing_rationale_when_labels_only():
    prediction = parse_classification_output("Labels: NORM")

    assert prediction.rationale == ""
    assert prediction.rationale_source == "missing"
    assert prediction.labels == [code for code in SCP_ALL_CODES if code in {"NORM"}]


def test_asl_block_contains_71_label_lines():
    prediction = PulsePrediction(
        labels=[SCP_ALL_CODES[0]],
        rationale="Example.",
        binary_vector=[1] + [0] * (NUM_CLASSES - 1),
        unknown_tokens=[],
        raw_output="Labels: test",
    )

    asl_block = prediction.as_asl_block()

    assert asl_block.startswith("<asl_labels>\n")
    assert f"{SCP_ALL_CODES[0]} 1" in asl_block
    assert asl_block.count("\n") >= NUM_CLASSES + 2
