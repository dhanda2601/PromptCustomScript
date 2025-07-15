import json
from typing import Dict, List, Any
from llm.llm_handler import run_llm_prompt
import logging

logger = logging.getLogger(__name__)

class PromptGenerator:
    def __init__(self, model_path: str, prompt_types: Dict[str, Any], business_rules: List[Dict[str, Any]]):
        self.model_path = model_path
        self.prompt_types = prompt_types
        self.business_rules = business_rules

    def get_all_prompt_types(self) -> List[str]:
        return list(self.prompt_types.keys())

    def get_prompt_info(self, prompt_type: str) -> Dict[str, Any]:
        return self.prompt_types.get(prompt_type, {})

    def validate_with_rules(self, variables: Dict[str, str], prompt_type: str) -> List[str]:
        errors = []
        relevant_rules = [rule for rule in self.business_rules if rule["Field"] in variables and prompt_type in rule.get("PromptType", [])]
        for rule in relevant_rules:
            field = rule["Field"]
            value = variables.get(field, "")
            rule_condition = rule["Rule"]
            error_message = rule["ErrorMessage"]
            
            try:
                if "is_string" in rule_condition and not isinstance(value, str):
                    errors.append(error_message)
                if "not_empty" in rule_condition and not value.strip():
                    errors.append(error_message)
                if "is_integer" in rule_condition:
                    try:
                        int_value = int(value)
                        if "value > 0" in rule_condition and int_value <= 0:
                            errors.append(error_message)
                    except ValueError:
                        errors.append(error_message)
                if "value in" in rule_condition:
                    allowed_values = rule_condition.split("value in ")[1].strip("[]").replace("'", "").split(", ")
                    if value not in allowed_values:
                        errors.append(error_message)
            except Exception as e:
                logger.error(f"Validation error for {field}: {e}")
                errors.append(f"Invalid {field}: {error_message}")
        
        return errors

    def generate_prompt(self, prompt_type: str, variables: Dict[str, str]) -> str:
        prompt_info = self.get_prompt_info(prompt_type)
        if not prompt_info:
            logger.error(f"Prompt type {prompt_type} not found")
            raise ValueError(f"Prompt type {prompt_type} not found")
        
        template = prompt_info.get("template", "")
        if not template:
            logger.error(f"No template found for prompt type {prompt_type}")
            raise ValueError(f"No template found for prompt type {prompt_type}")
        
        # Add prompt type prefix for relevant prompt types
        structured_types = [
            "Data-Mapping", "Chain-of-Thought", "Self-Consistency", "Tree-of-Thoughts",
            "Graph-of-Thought", "Skeleton-of-Thought", "Chain-of-Verification", "ReAct",
            "Recursive Prompting", "Automatic Prompt Engineer (APE)", "Automatic Reasoning and Tool-use (ART)",
            "Chain-of-Note", "Chain-of-Code", "Chain-of-Symbol", "Structured Chain-of-Thought",
            "Contrastive Chain-of-Thought", "Logical Chain-of-Thought", "System 2 Attention Prompting",
            "Research-Challenges", "Data Quality", "Self-Healing", "Code Conversion",
            "Presales", "HR Operations", "Learning and Knowledge", "Finance", "Project Management",
            "Instruction Prompting and Tuning"
        ]
        if prompt_type in structured_types:
            template = f"Prompt Type: {prompt_type}. {template}"
        
        try:
            # Replace placeholders with variable values
            final_prompt = template
            for var, value in variables.items():
                placeholder = "{" + var + "}"
                final_prompt = final_prompt.replace(placeholder, value)
            logger.info(f"Generated prompt for {prompt_type}: {final_prompt[:200]}...")
            return final_prompt
        except Exception as e:
            logger.error(f"Failed to generate prompt for {prompt_type}: {e}")
            raise ValueError(f"Failed to generate prompt: {e}")

    async def get_llm_response(self, prompt: str, parse_json: bool = False) -> Any:
        try:
            response = await run_llm_prompt([prompt], parse_json=parse_json)
            return response[0]
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")
            raise