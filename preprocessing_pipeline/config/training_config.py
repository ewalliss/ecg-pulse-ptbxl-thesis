"""
Training configuration for the ECG-LLaVA pipeline.

Centralizes all training hyperparameters from thesis ch3 SS3.2.
Extends the project convention established by preprocessing_config.py.
"""

from pathlib import Path
import os

from preprocessing_pipeline.config.preprocessing_config import OUTPUT_ROOT, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD

# ---------------------------------------------------------------------------
# Dataset output paths
# ---------------------------------------------------------------------------

INSTRUCTION_DIR = OUTPUT_ROOT / "instruction"
"""Root directory for all generated instruction dataset JSONs."""

STAGE1_TRAIN_JSON = INSTRUCTION_DIR / "stage1_train.json"
STAGE1_VAL_JSON = INSTRUCTION_DIR / "stage1_val.json"
STAGE2_TRAIN_JSON = INSTRUCTION_DIR / "stage2_train.json"
STAGE2_VAL_JSON = INSTRUCTION_DIR / "stage2_val.json"

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

LLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
CLIP_MODEL_NAME: str = "openai/clip-vit-large-patch14-336"

# ---------------------------------------------------------------------------
# Stage 1: Feature alignment (thesis ch3 lines 411-416)
# ---------------------------------------------------------------------------

STAGE1_EPOCHS: int = 1
STAGE1_BATCH_SIZE: int = 256
STAGE1_LEARNING_RATE: float = 1e-3
STAGE1_OPTIMIZER: str = "AdamW"
STAGE1_LR_SCHEDULER: str = "cosine"
STAGE1_PRECISION: str = "bf16"          # BFloat16 for projector

# ---------------------------------------------------------------------------
# Stage 2: QLoRA fine-tuning (thesis ch3 lines 418-423)
# ---------------------------------------------------------------------------

STAGE2_EPOCHS: int = 3
STAGE2_BATCH_SIZE: int = 64
STAGE2_LEARNING_RATE: float = 2e-4
STAGE2_OPTIMIZER: str = "AdamW"
STAGE2_LR_SCHEDULER: str = "cosine"
STAGE2_WARMUP_RATIO: float = 0.05      # 5% warmup
STAGE2_PRECISION: str = "bf16"

# ---------------------------------------------------------------------------
# QLoRA configuration (thesis ch3 SS3.2, lines 210-274)
# ---------------------------------------------------------------------------

QLORA_BITS: int = 4                     # NF4 quantization
QLORA_DOUBLE_QUANT: bool = True         # ~0.37 bits/param savings
LORA_RANK: int = 16
LORA_ALPHA: int = 32                    # scaling = alpha/r = 2
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# ASL-Gen loss hyperparameters (thesis ch3 SS3.2, lines 343-374)
# ---------------------------------------------------------------------------

ASLGEN_LAMBDA_0: float = 0.3           # balance prefix vs label loss
ASLGEN_GAMMA_POS: float = 1.0          # focusing for positives
ASLGEN_GAMMA_NEG: float = 4.0          # focusing for negatives
ASLGEN_PROB_SHIFT_M: float = 0.05      # probability shift margin

# ---------------------------------------------------------------------------
# Instruction dataset parameters
# ---------------------------------------------------------------------------

NUM_SCP_CODES: int = 44                 # K = 44 SCP-ECG diagnostic codes
