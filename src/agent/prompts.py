"""System prompts and templates for the MDM transformation agent."""

from pathlib import Path
import yaml


def load_system_prompt() -> str:
    """Load the system prompt from settings.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
    
    if not config_path.exists():
        return DEFAULT_SYSTEM_PROMPT
    
    try:
        with open(config_path, "r") as f:
            settings = yaml.safe_load(f)
            return settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT).strip()
    except Exception:
        return DEFAULT_SYSTEM_PROMPT


DEFAULT_SYSTEM_PROMPT = """You are a Python data transformation expert. Generate clean pandas code.

Requirements:
1. Function signature: def transform(df_dict: dict) -> pd.DataFrame
2. Extract target dataset from df_dict by name
3. Return transformed DataFrame
4. Include error handling
5. Use pandas idioms (dropna, merge, groupby, etc.)
6. Add explanatory comments
7. No external imports (os, sys, subprocess, open)
8. Validate inputs and outputs

Output only the Python code block in triple backticks."""


def build_code_generation_prompt(
    user_prompt: str, 
    target_dataset: str, 
    columns: list[str],
    sample_data: str = ""
) -> str:
    """Build the full prompt for code generation with context."""
    system_prompt = load_system_prompt()
    
    context = f"""Target Dataset: {target_dataset}
Available Columns: {', '.join(columns)}"""
    
    if sample_data:
        context += f"\nSample Data (first row):\n{sample_data}"
    
    return f"""{system_prompt}

{context}

User Request: {user_prompt}

Generate the transformation code now:"""
