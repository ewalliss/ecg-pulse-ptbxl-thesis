from preprocessing_pipeline.config.preprocessing_config import SCP_ALL_CODES
from preprocessing_pipeline.pulse_ptbxl.build_stage2_dataset import build_stage2_sample, format_binary_answer


def test_format_binary_answer_contains_all_71_labels():
    vector = [0] * len(SCP_ALL_CODES)
    vector[0] = 1

    answer = format_binary_answer(SCP_ALL_CODES, vector)

    assert answer.startswith("<asl_labels>\n")
    assert answer.count("\n") >= len(SCP_ALL_CODES) + 2
    assert f"{SCP_ALL_CODES[0]} 1" in answer
    assert f"Positive SCP-ECG labels: {SCP_ALL_CODES[0]}" in answer


def test_build_stage2_sample_matches_pulse_conversation_schema():
    vector = [0] * len(SCP_ALL_CODES)
    vector[0] = 1
    vector[3] = 1

    sample = build_stage2_sample(
        ecg_id=7,
        fold=1,
        image_name="00007.png",
        label_codes=SCP_ALL_CODES,
        scp_vector=vector,
    )

    assert sample["id"] == "ptbxl_00007"
    assert sample["image"] == "00007.png"
    assert sample["fold"] == 1
    assert sample["ecg_id"] == 7
    assert sample["scp_vector"] == vector
    assert sample["scp_codes"] == [SCP_ALL_CODES[0], SCP_ALL_CODES[3]]
    assert sample["conversations"][0]["from"] == "human"
    assert sample["conversations"][0]["value"].startswith("<image>\n")
    assert sample["conversations"][1]["from"] == "gpt"
    assert "<asl_labels>" in sample["conversations"][1]["value"]
