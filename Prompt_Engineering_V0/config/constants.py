import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

# ----- Base Directory -----
BASE_DIR = Path(__file__).parent.parent

# ----- Environment Configuration -----
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NO_PROXY'] = '*'

PAGE_TITLE = "🧠 GenAI-Powered 📘 RAG Architecture - Multi-Format Vector QA"

# ----- Key Directories -----
CONFIG_DIR = BASE_DIR / "config" / "json_configs"
PROMPT_FILE = CONFIG_DIR / "Prompts_Types.json"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
MODELS_DIR = Path(r"C:\Users\VRamamurthy\OneDrive - Mastech Digital\GenAI-Developers\modelslist")

# ----- Model Auto-Discovery -----
def discover_models() -> Tuple[Dict[str, Dict], str, str]:
    model_registry = {}
    default_llm = None
    default_embedding = None

    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        model_type = None

        if "mini" in model_name.lower() or "embedding" in model_name.lower():
            model_type = "embedding"
            if default_embedding is None:
                default_embedding = model_name
        elif any(x in model_name.lower() for x in ["llama", "phi", "deepseek", "coder"]):
            model_type = "text-generation"
            if default_llm is None and "deepseek-1_3b" in model_name.lower():
                default_llm = model_name

        if model_type:
            model_registry[model_name] = {
                "path": model_dir,
                "type": model_type,
                "default": (model_name == default_llm) if model_type == "text-generation" else (model_name == default_embedding)
            }

    if not default_llm:
        default_llm = "deepseek-1_3b-gguf"
        model_registry.setdefault(default_llm, {
            "path": MODELS_DIR / "deepseek-1_3b-gguf",
            "type": "text-generation",
            "default": True
        })

    if not default_embedding:
        default_embedding = "all-MiniLM-L6-v2"
        model_registry.setdefault(default_embedding, {
            "path": MODELS_DIR / "all-MiniLM-L6-v2",
            "type": "embedding",
            "default": True
        })

    return model_registry, default_llm, default_embedding

# Discover models
MODEL_PATHS, DEFAULT_LLM_MODEL, EMBEDDING_MODEL = discover_models()

# ----- Cache Configuration -----
onedrive_cache = Path(r"C:\Users\VRamamurthy\OneDrive - Mastech Digital\GenAI-Developers\model_cache")
MODEL_CACHE_DIR = onedrive_cache if onedrive_cache.exists() else BASE_DIR / "model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ----- Constants -----
EMBEDDING_FUNCTION = "sentence_transformer"
EMBEDDING_DIMENSION = 384
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
MODEL_CACHE_EXPIRY_DAYS = 90

# -------- Chunck Size and Overlap --------
CHUNK_SIZE = 512
CHUNK_OVERLAP_LEN = 50

# ----- Type Registry -----
MODEL_TYPES = {name: props["type"] for name, props in MODEL_PATHS.items()}

# ----- Cache Policy Class -----
class CachePolicy:
    @staticmethod
    def should_cache(model_name: str) -> bool:
        return MODEL_PATHS.get(model_name, {}).get("path", Path("_null")).exists()

    @staticmethod
    def get_cache_path(model_name: str) -> Path:
        return MODEL_CACHE_DIR / f"{model_name.replace('/', '_')}.cache"

    @staticmethod
    def is_cache_valid(cache_path: Path) -> bool:
        return cache_path.exists() and cache_path.stat().st_size > 0

# ----- Utility Functions -----
def get_available_models(model_type: str = None) -> List[str]:
    if model_type:
        return [name for name, props in MODEL_PATHS.items() if props["type"] == model_type]
    return list(MODEL_PATHS.keys())

def get_default_model(model_type: str) -> Optional[str]:
    for name, props in MODEL_PATHS.items():
        if props["type"] == model_type and props.get("default", False):
            return name
    return None
# ----- Debug Utilities -----
class DebugUtils:
    @staticmethod
    def check_cache_file(model_name: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
        """Check structure and metadata of a cached model."""
        cache_path = CachePolicy.get_cache_path(model_name)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return {
                'exists': True,
                'valid_structure': isinstance(data, dict),
                'has_model_key': 'model' in data,
                'timestamp': data.get('timestamp'),
                'version': data.get('version'),
                'size_mb': cache_path.stat().st_size / (1024 ** 2),
                'raw_data_keys': list(data.keys()) if isinstance(data, dict) else None,
                'last_modified': os.path.getmtime(cache_path)
            }
        except Exception as e:
            return {'exists': True, 'error': str(e), 'valid': False}

    @staticmethod
    def format_bytes(size_bytes: int) -> str:
        """Format byte size to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"

# ----- Ensure Required Directories -----
for directory in [CONFIG_DIR, CHROMA_DB_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
