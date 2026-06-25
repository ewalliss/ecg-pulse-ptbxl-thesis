"""
Translate PTB-XL German clinical reports → English (offline MarianMT, cached).

PTB-XL ships the ``report`` column as German clinical shorthand
(e.g. ``"sinusrhythmus normales ekg"``).  PULSE-7B is an English VLM, so we
machine-translate the reports once to English and cache the result.  The
translated English report becomes the per-record "Interpretation" prefix of
the Stage-2 instruction answer — a real chain-of-thought grounded in the
original cardiologist text, instead of a fixed (and clinically wrong) template.

Design notes
------------
- **Dedupe before translating.** Most reports repeat verbatim thousands of
  times (``"sinusrhythmus normales ekg"`` alone covers a large fraction).  We
  translate only the *unique* cleaned strings, then map back per ``ecg_id``.
- **Cache to JSON** keyed by ``ecg_id``.  Re-runs skip translation entirely.
- **Deterministic** decoding (beam search, no sampling) → reproducible.
- Model: ``Helsinki-NLP/opus-mt-de-en`` (loaded lazily, only when needed).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from preprocessing_pipeline.utils.logger import get_logger

log = get_logger(__name__)

_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
_MAX_TOKENS = 128
_NUM_BEAMS = 4


def _clean_german(text: object) -> str:
    """Strip the stray leading-quote artifact and collapse whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = str(text).replace('"', " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _sentence_case(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _load_translator():
    from transformers import MarianMTModel, MarianTokenizer
    import torch

    tokenizer = MarianTokenizer.from_pretrained(_MODEL_NAME)
    model = MarianMTModel.from_pretrained(_MODEL_NAME)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def _translate_unique(strings: list[str], batch_size: int = 64) -> dict[str, str]:
    """Translate a list of unique German strings → dict[de] = en."""
    import torch

    tokenizer, model = _load_translator()
    device = next(model.parameters()).device
    out: dict[str, str] = {}
    for i in range(0, len(strings), batch_size):
        chunk = strings[i : i + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=_MAX_TOKENS
        ).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=_MAX_TOKENS, num_beams=_NUM_BEAMS)
        for de, en in zip(chunk, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[de] = _sentence_case(en.strip())
    return out


def build_translated_report_map(
    metadata_df: pd.DataFrame,
    cache_path: Path,
) -> dict[int, str]:
    """Return ``{ecg_id: english_report}`` for every record in ``metadata_df``.

    Translates only the records not already in the on-disk cache, and only the
    *unique* German strings among those (deduplicated).  Writes the merged
    cache back to ``cache_path``.

    Parameters
    ----------
    metadata_df:
        Filtered PTB-XL metadata; must contain ``ecg_id`` and ``report``.
    cache_path:
        JSON file ``{ecg_id: english_report}``.  Created if absent.
    """
    cache_path = Path(cache_path)
    cache: dict[int, str] = {}
    if cache_path.exists():
        cache = {int(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}

    # Collect records missing from cache, cleaned.
    pending: dict[int, str] = {}
    for _, row in metadata_df.iterrows():
        ecg_id = int(row["ecg_id"])
        if ecg_id in cache:
            continue
        pending[ecg_id] = _clean_german(row.get("report", ""))

    if pending:
        unique_de = sorted({de for de in pending.values() if de})
        log.info(
            "Translating %d unique German reports (covering %d uncached records) → English...",
            len(unique_de),
            len(pending),
        )
        de_to_en = _translate_unique(unique_de) if unique_de else {}
        for ecg_id, de in pending.items():
            cache[ecg_id] = de_to_en.get(de, "") if de else ""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8"
        )
        log.info("Cached %d translated reports → %s", len(cache), cache_path)
    else:
        log.info("All %d reports already translated (cache hit) → %s", len(cache), cache_path)

    return cache
