import os
from pathlib import Path
from typing import Final
import platform

# Base directory relative to this file (config/config.py)
BASE_DIR: Final[Path] = Path(__file__).resolve().parent  # Resolves to Prompt_Engineering\config

# Directories
DATA_DIR: Final[Path] = BASE_DIR.parent / "data"  # Resolves to Prompt_Engineering\data
INPUTS_DIR: Final[Path] = BASE_DIR.parent / "inputs"  # Resolves to Prompt_Engineering\inputs
MAPPINGS_DIR: Final[Path] = DATA_DIR / "mappings"  # Resolves to Prompt_Engineering\data\mappings

# Dynamic model directory based on user home
if platform.system() == "Windows":
    default_path_str = r"%USERPROFILE%\code\modelslist"
else:
    default_path_str = r"$HOME/code/modelslist"

# Expand environment variables and user home
expanded_path_str = os.path.expandvars(default_path_str)
expanded_path_str = os.path.expanduser(expanded_path_str)
MODELS_DIR: Final[Path] = Path(expanded_path_str).resolve()  # Resolves to C:\Users\VRamamurthy\code\modelslist

CONFIG_DIR: Final[Path] = BASE_DIR / "json_configs"  # Resolves to Prompt_Engineering\config\json_configs
PROMPT_TYPES_FILE: Final[Path] = CONFIG_DIR / "Prompts_Types.json"
BUSINESS_RULES_FILE: Final[Path] = CONFIG_DIR / "BusinessRules.json"

# Model path
DEEPSEEK_MODEL: Final[Path] = MODELS_DIR / "deepseek-1_3b-gguf" / "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"

# LLM settings
LLM_TEMPERATURE: Final[float] = 0.6
LLM_MAX_TOKENS: Final[int] = 2048
LLM_CONTEXT_SIZE: Final[int] = 4096
LLM_BATCH_SIZE: Final[int] = 2
LLM_NUM_THREADS: Final[int] = 4
LLM_N_GPU_LAYERS: Final[int] = 0
LLM_REPEAT_PENALTY: Final[float] = 1.1
LLM_TOP_P: Final[float] = 0.9

PAGE_TITLE: Final[str] = "GenAI Prompt Template Generation"

def validate_paths_and_models() -> None:
    """Ensure directories exist and required files are present."""
    for directory in [DATA_DIR, INPUTS_DIR, MAPPINGS_DIR, MODELS_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    missing = [str(f) for f in [PROMPT_TYPES_FILE, BUSINESS_RULES_FILE, DEEPSEEK_MODEL] if not f.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))

# Validate immediately on import
validate_paths_and_models()