import importlib.util
import sys
import types
from pathlib import Path

import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "PULSE/LLaVA/llava/train/train.py"


def _install_llava_stubs():
    constants = types.ModuleType("llava.constants")
    constants.IGNORE_INDEX = -100
    constants.IMAGE_TOKEN_INDEX = -200
    constants.DEFAULT_IMAGE_TOKEN = "<image>"
    constants.DEFAULT_IM_START_TOKEN = "<im_start>"
    constants.DEFAULT_IM_END_TOKEN = "<im_end>"

    trainer = types.ModuleType("llava.train.llava_trainer")
    trainer.LLaVATrainer = object

    conversation = types.ModuleType("llava.conversation")
    conversation_lib = types.SimpleNamespace(
        SeparatorStyle=types.SimpleNamespace(PLAIN="plain", LLAMA_2="llama_2", TWO="two", MPT="mpt"),
        default_conversation=types.SimpleNamespace(sep_style="plain", version="plain"),
        conv_templates={},
    )
    conversation.SeparatorStyle = conversation_lib.SeparatorStyle
    conversation.default_conversation = conversation_lib.default_conversation
    conversation.conv_templates = conversation_lib.conv_templates

    model = types.ModuleType("llava.model")
    mm_utils = types.ModuleType("llava.mm_utils")
    mm_utils.tokenizer_image_token = lambda *args, **kwargs: torch.tensor([1])
    mm_utils.process_anyres_image = lambda image, processor, pinpoints: image

    sys.modules.setdefault("llava", types.ModuleType("llava"))
    sys.modules["llava.constants"] = constants
    sys.modules.setdefault("llava.train", types.ModuleType("llava.train"))
    sys.modules["llava.train.llava_trainer"] = trainer
    sys.modules["llava.conversation"] = conversation
    sys.modules["llava.model"] = model
    sys.modules["llava.mm_utils"] = mm_utils


def _load_train_module():
    _install_llava_stubs()
    spec = importlib.util.spec_from_file_location("pulse_train", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyTokenizer:
    pad_token_id = 0
    model_max_length = 5


def test_collator_batches_asl_fields():
    train_module = _load_train_module()
    collator = train_module.DataCollatorForSupervisedDataset(tokenizer=DummyTokenizer())
    instances = [
        {
            "input_ids": torch.tensor([3, 4, 5]),
            "labels": torch.tensor([-100, 4, 5]),
            "asl_targets": torch.tensor([1.0, 0.0]),
            "asl_label_positions": torch.tensor([1, 2]),
        },
        {
            "input_ids": torch.tensor([6, 7]),
            "labels": torch.tensor([-100, 7]),
            "asl_targets": torch.tensor([0.0, 1.0]),
            "asl_label_positions": torch.tensor([1, -1]),
        },
    ]

    batch = collator(instances)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert torch.equal(batch["asl_targets"], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(batch["asl_label_positions"], torch.tensor([[1, 2], [1, -1]]))


def test_find_asl_label_positions_scans_supervised_tokens_in_order():
    train_module = _load_train_module()
    labels = torch.tensor([-100, 10, 20, -100, 10, 30])
    positions = train_module._find_asl_label_positions(labels, [10, 30, 99])

    assert torch.equal(positions, torch.tensor([0, 3, -1]))
