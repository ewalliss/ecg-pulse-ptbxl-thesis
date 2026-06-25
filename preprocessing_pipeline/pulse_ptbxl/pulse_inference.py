from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import re
import sys
from typing import Any
import warnings

from PIL import Image
import torch

from preprocessing_pipeline.config.preprocessing_config import SCP_ALL_CODES
from preprocessing_pipeline.preprocessing.label.multilabel_encoder import (
    encode_scp_codes_to_binary_vector,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLAVA_ROOT = _REPO_ROOT / "source" / "LLaVA"
_CANONICAL_CODES = frozenset(SCP_ALL_CODES)
_LABEL_SCAN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(code) for code in sorted(SCP_ALL_CODES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_SECTION_TEMPLATE = r"(?ims)^\s*{name}\s*:\s*(.+?)(?=^\s*[A-Za-z ]+\s*:|\Z)"
_LABEL_SPLIT_PATTERN = re.compile(r"[,;|\n]+")
_LABELS_HEADER_PATTERN = re.compile(r"(?im)^\s*Labels\s*:\s*")
_RATIONALE_HEADERS = ("Rationale", "Reasoning", "Explanation")
_NULL_LABEL_TOKENS = {"NONE", "IDONTKNOW"}
_CONFIDENCE_PATTERN = re.compile(r"([A-Z0-9]+)\s*\(([0-9.]+)\)")


@dataclass(frozen=True)
class PulsePrediction:
    labels: list[str]
    rationale: str
    binary_vector: list[int]
    unknown_tokens: list[str]
    raw_output: str
    prompt: str = ""
    rationale_source: str = "missing"
    confidence_scores: dict[str, float] | None = None

    def as_asl_block(self) -> str:
        lines = ["<asl_labels>"]
        positives = []
        for code, bit in zip(SCP_ALL_CODES, self.binary_vector):
            lines.append(f"{code} {bit}")
            if bit == 1:
                positives.append(code)
        lines.append("</asl_labels>")
        lines.append(f"Positive SCP-ECG labels: {', '.join(positives) if positives else 'NORMAL'}")
        return "\n".join(lines)


@dataclass
class LoadedPulseClassifier:
    tokenizer: Any
    model: Any
    image_processor: Any
    conv_templates: Any
    tokenizer_image_token: Any
    process_images: Any
    image_token_index: int
    default_image_token: str
    default_im_start_token: str
    default_im_end_token: str
    conv_mode: str

    def classify_image(
        self,
        image_file: str | Path,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
        custom_prompt: str | None = None,
    ) -> PulsePrediction:
        image = Image.open(image_file).convert("RGB")
        query = custom_prompt if custom_prompt else build_classification_prompt()
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = (
                f"{self.default_im_start_token}"
                f"{self.default_image_token}"
                f"{self.default_im_end_token}\n{query}"
            )
        else:
            qs = f"{self.default_image_token}\n{query}"

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0)
        model_device = _get_model_device(self.model)
        input_ids = input_ids.to(model_device)

        image_tensor = self.process_images(
            [image],
            self.image_processor,
            self.model.config,
        )[0]
        image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
        image_tensor = image_tensor.unsqueeze(0).to(device=model_device, dtype=image_dtype)

        generate_kwargs = {
            "images": image_tensor,
            "image_sizes": [image.size],
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                **generate_kwargs,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        decoded_ids = generated_ids if generated_ids.shape[1] > 0 else output_ids
        raw_output = self.tokenizer.batch_decode(
            decoded_ids,
            skip_special_tokens=True,
        )[0].strip()
        return parse_classification_output(raw_output, prompt=prompt)


def build_classification_prompt() -> str:
    label_vocab = ", ".join(SCP_ALL_CODES)
    return (
        "You are an expert cardiologist analyzing an ECG image. "
        "Classify this ECG with PTB-XL SCP-ECG diagnostic codes. This is a multi-label task.\n\n"
        "Provide your clinical diagnosis in exactly three lines:\n"
        "1. Rationale: Explain the visual ECG features you observe (waveform morphology, intervals, segments) "
        "and your diagnostic reasoning for why you chose each label. Be specific about clinical evidence.\n"
        "2. Labels: List the SCP codes with confidence scores in format: CODE1 (0.XX), CODE2 (0.XX), ...\n"
        "   Confidence scale: 0.90-1.00 (definitive), 0.70-0.89 (probable), 0.50-0.69 (possible)\n"
        "3. If no diagnostic code applies with confidence ≥0.50, output: Labels: NONE\n\n"
        f"Allowed SCP codes: {label_vocab}\n\n"
        "Requirements:\n"
        "- Always provide clinical rationale first\n"
        "- Only use codes from the allowed list\n"
        "- Include confidence score (0.00-1.00) for each label\n"
        "- Base confidence on strength of visual evidence\n"
        "- If uncertain, explain why in rationale and use lower confidence"
    )


def parse_classification_output(
    raw_output: str,
    prompt: str = "",
) -> PulsePrediction:
    rationale, rationale_source = _extract_rationale(raw_output)
    labels_section = _extract_section(raw_output, "Labels")
    labels, unknown_tokens, confidence_scores = _parse_labels_section(labels_section)
    if not labels and not labels_section:
        labels = _scan_labels(raw_output)

    binary_vector = encode_scp_codes_to_binary_vector(labels).astype(int).tolist()
    return PulsePrediction(
        labels=labels,
        rationale=rationale,
        binary_vector=binary_vector,
        unknown_tokens=unknown_tokens,
        raw_output=raw_output.strip(),
        prompt=prompt,
        rationale_source=rationale_source,
        confidence_scores=confidence_scores if confidence_scores else None,
    )


def load_pulse_classifier(
    model_path: str = "PULSE-ECG/PULSE-7B",
    device: str = "cuda",
    conv_mode: str = "llava_v1",
    load_4bit: bool = False,
) -> LoadedPulseClassifier:
    llava = _load_llava_modules()
    llava["disable_torch_init"]()

    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    model_name = llava["get_model_name_from_path"](model_path)
    tokenizer, model, image_processor, _ = llava["load_pretrained_model"](
        model_path,
        None,
        model_name,
        load_4bit=load_4bit,
        device_map="auto",
        device=device,
    )
    if image_processor is None:
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and hasattr(vision_tower, "image_processor"):
            image_processor = vision_tower.image_processor
    return LoadedPulseClassifier(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        conv_templates=llava["conv_templates"],
        tokenizer_image_token=llava["tokenizer_image_token"],
        process_images=llava["process_images"],
        image_token_index=llava["IMAGE_TOKEN_INDEX"],
        default_image_token=llava["DEFAULT_IMAGE_TOKEN"],
        default_im_start_token=llava["DEFAULT_IM_START_TOKEN"],
        default_im_end_token=llava["DEFAULT_IM_END_TOKEN"],
        conv_mode=conv_mode,
    )


def _extract_section(text: str, name: str) -> str:
    match = re.search(_SECTION_TEMPLATE.format(name=re.escape(name)), text)
    if not match:
        return ""
    return " ".join(line.strip() for line in match.group(1).splitlines() if line.strip())


def _extract_rationale(text: str) -> tuple[str, str]:
    for header in _RATIONALE_HEADERS:
        rationale = _extract_section(text, header)
        if rationale:
            return rationale, "exact"

    labels_match = _LABELS_HEADER_PATTERN.search(text)
    if labels_match:
        pre_labels = text[:labels_match.start()].strip()
        fallback = _strip_known_headers(pre_labels)
        if fallback:
            return fallback, "fallback"
        if labels_match.start() == 0:
            return "", "missing"

    stripped = _strip_known_headers(text).strip()
    if stripped and not _LABELS_HEADER_PATTERN.search(stripped):
        return stripped, "fallback"

    return "", "missing"


def _parse_labels_section(section: str) -> tuple[list[str], list[str], dict[str, float]]:
    if not section:
        return [], [], {}

    labels: set[str] = set()
    unknown_tokens: list[str] = []
    seen_unknown: set[str] = set()
    confidence_scores: dict[str, float] = {}

    for token in _LABEL_SPLIT_PATTERN.split(section):
        token = token.strip()
        if not token:
            continue

        conf_match = _CONFIDENCE_PATTERN.match(token)
        if conf_match:
            code = conf_match.group(1).upper()
            try:
                confidence = float(conf_match.group(2))
                if code in _CANONICAL_CODES:
                    labels.add(code)
                    confidence_scores[code] = confidence
                    continue
            except ValueError:
                pass

        normalized = _normalize_label_token(token)
        if not normalized or normalized in _NULL_LABEL_TOKENS:
            continue
        if normalized in _CANONICAL_CODES:
            labels.add(normalized)
            continue
        if normalized not in seen_unknown:
            unknown_tokens.append(normalized)
            seen_unknown.add(normalized)

    return _canonicalize_labels(labels), unknown_tokens, confidence_scores


def _scan_labels(text: str) -> list[str]:
    matches = {match.upper() for match in _LABEL_SCAN_PATTERN.findall(text)}
    return _canonicalize_labels(matches)


def _canonicalize_labels(labels: set[str] | frozenset[str]) -> list[str]:
    return [code for code in SCP_ALL_CODES if code in labels]


def _normalize_label_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", token).upper()


def _strip_known_headers(text: str) -> str:
    stripped = text.strip()
    for header in _RATIONALE_HEADERS:
        pattern = re.compile(rf"(?is)^\s*{re.escape(header)}\s*:\s*")
        stripped = pattern.sub("", stripped)
    stripped = _LABELS_HEADER_PATTERN.sub("", stripped)
    return stripped.strip()


def _get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _load_llava_modules() -> dict[str, Any]:
    if str(_LLAVA_ROOT) not in sys.path:
        sys.path.insert(0, str(_LLAVA_ROOT))

    constants = import_module("llava.constants")
    conversation = import_module("llava.conversation")
    model_builder = import_module("llava.model.builder")
    mm_utils = import_module("llava.mm_utils")
    utils = import_module("llava.utils")

    return {
        "IMAGE_TOKEN_INDEX": constants.IMAGE_TOKEN_INDEX,
        "DEFAULT_IMAGE_TOKEN": constants.DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_START_TOKEN": constants.DEFAULT_IM_START_TOKEN,
        "DEFAULT_IM_END_TOKEN": constants.DEFAULT_IM_END_TOKEN,
        "conv_templates": conversation.conv_templates,
        "load_pretrained_model": model_builder.load_pretrained_model,
        "process_images": mm_utils.process_images,
        "tokenizer_image_token": mm_utils.tokenizer_image_token,
        "get_model_name_from_path": mm_utils.get_model_name_from_path,
        "disable_torch_init": utils.disable_torch_init,
    }
