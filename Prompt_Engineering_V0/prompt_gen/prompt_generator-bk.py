import streamlit as st
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import logging
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Enums and Data Classes ---
class PromptCategory(Enum):
    GENERAL = "General"
    REASONING = "Reasoning"
    CREATIVE = "Creative"
    TECHNICAL = "Technical"
    ANALYTICAL = "Analytical"
    INTERACTIVE = "Interactive"

@dataclass
class BusinessRule:
    field: str
    name: str
    rule_type: str
    expected_value: Optional[Union[str, int, float, bool]]
    description: Optional[str] = None
    severity: Optional[str] = "error"

@dataclass
class PromptConfig:
    template: str
    input_vars: List[str]
    description: str
    category: PromptCategory
    examples: Optional[List[Dict[str, str]]] = None
    version: Optional[str] = "1.0"
    tags: Optional[List[str]] = None

# --- Prompt Generator ---
class PromptGenerator:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._validate_config_path()
        self.prompt_configs: Dict[str, PromptConfig] = {}
        self.business_rules: List[BusinessRule] = []
        self.prompt_templates: Dict[str, ChatPromptTemplate] = {}
        self._load_configurations()
        self._build_templates()

    def _validate_config_path(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_path}")
        if not self.config_path.is_dir():
            raise NotADirectoryError(f"Config path is not a directory: {self.config_path}")

    def _load_configurations(self) -> None:
        try:
            prompts_path = self.config_path / "Prompts_Types.json"
            if prompts_path.exists():
                raw_prompts = self._load_json_file(prompts_path)
                self.prompt_configs = self._parse_prompt_configs(raw_prompts)
            rules_path = self.config_path / "BusinessRules.json"
            if rules_path.exists():
                raw_rules = self._load_json_file(rules_path)
                self.business_rules = self._parse_business_rules(raw_rules)
        except Exception as e:
            logger.error(f"Failed to load configurations: {str(e)}")
            raise

    def _load_json_file(self, file_path: Path) -> Union[Dict, List]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_prompt_configs(self, raw_configs: Dict) -> Dict[str, PromptConfig]:
        parsed = {}
        for name, config in raw_configs.items():
            try:
                category_str = config.get("category", "GENERAL").upper()
                category = PromptCategory[category_str] if category_str in PromptCategory.__members__ else PromptCategory.GENERAL
                parsed[name] = PromptConfig(
                    template=config["template"],
                    input_vars=config.get("input_vars", []),
                    description=config.get("description", ""),
                    category=category,
                    examples=config.get("examples"),
                    version=config.get("version", "1.0"),
                    tags=config.get("tags", [])
                )
            except Exception as e:
                logger.error(f"Error parsing prompt '{name}': {str(e)}")
        return parsed

    def _parse_business_rules(self, raw_rules: List) -> List[BusinessRule]:
        parsed = []
        for rule in raw_rules:
            try:
                parsed.append(BusinessRule(
                    field=rule["Field"],
                    name=rule["Rule Name"],
                    rule_type=rule["Rule Type"],
                    expected_value=rule.get("Expected Value"),
                    description=rule.get("Description"),
                    severity=rule.get("Severity", "error")
                ))
            except Exception as e:
                logger.error(f"Error parsing business rule: {str(e)}")
        return parsed

    def _build_templates(self) -> None:
        for name, config in self.prompt_configs.items():
            try:
                self.prompt_templates[name] = ChatPromptTemplate.from_messages([
                    HumanMessagePromptTemplate.from_template(config.template)
                ])
            except Exception as e:
                logger.error(f"Failed to build template for '{name}': {str(e)}")

    def get_all_prompt_types(self, category: Optional[PromptCategory] = None) -> List[str]:
        if category:
            return [name for name, config in self.prompt_configs.items() if config.category == category]
        return list(self.prompt_configs.keys())

    def get_prompt_info(self, prompt_type: str) -> Optional[Dict]:
        config = self.prompt_configs.get(prompt_type)
        if not config:
            return None
        return {
            "name": prompt_type,
            "description": config.description,
            "category": config.category.value,
            "input_variables": config.input_vars,
            "examples": config.examples,
            "version": config.version,
            "tags": config.tags
        }

    def get_prompt_categories(self) -> List[Tuple[str, str]]:
        return [(cat.name, cat.value) for cat in PromptCategory]

    def generate_prompt(self, prompt_type: str, variables: Dict[str, Union[str, int, float]]) -> Optional[str]:
        try:
            prompt = self.prompt_templates[prompt_type].format_messages(**variables)
            return str(prompt[0].content)
        except Exception as e:
            logger.error(f"Error generating prompt '{prompt_type}': {str(e)}")
            return None

    def get_business_rules(self, field: Optional[str] = None) -> List[BusinessRule]:
        return [r for r in self.business_rules if r.field == field] if field else self.business_rules

    def validate_with_rules(self, data: Dict, rules: Optional[List[BusinessRule]] = None) -> Dict[str, List[str]]:
        errors = {}
        rules_to_check = rules if rules else self.business_rules
        for rule in rules_to_check:
            if rule.field not in data:
                if rule.severity == "error":
                    errors.setdefault(rule.field, []).append(f"Missing required field: {rule.name}")
                continue
            value = data[rule.field]
            if rule.rule_type == "NotNull" and (value is None or value == ""):
                errors.setdefault(rule.field, []).append(f"Field cannot be null: {rule.name}")
            elif rule.rule_type == "Equals" and value != rule.expected_value:
                errors.setdefault(rule.field, []).append(f"Must equal {rule.expected_value}: {rule.name}")
        return errors