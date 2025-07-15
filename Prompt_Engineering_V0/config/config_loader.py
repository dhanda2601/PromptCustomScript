import json
from pathlib import Path
from config.constants import PROMPT_FILE

class ConfigLoader:
    @staticmethod
    def load_prompts_config() -> dict:
        try:
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
