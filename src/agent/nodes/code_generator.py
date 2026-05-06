import re
from src.core.logging.logger import get_logger
from src.agent.prompts import build_code_generation_prompt

logger = get_logger(__name__)


def _extract_code_block(response_text: str) -> str:
    """Extract Python code from markdown code blocks in LLM response."""
    # Try to extract code from ```python ... ``` blocks
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to extract from ``` ... ``` blocks (generic)
    match = re.search(r"```\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code blocks found, assume the entire response is code
    return response_text.strip()


def _create_llm_instance(provider: str, model: str):
    """Create an LLM instance based on provider."""
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=model, temperature=0.3, timeout=120)
        except ImportError as e:
            logger.error(f"langchain-ollama not installed: {e}")
            raise
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=0.3, timeout=120)
        except ImportError as e:
            logger.error(f"langchain-google-genai not installed: {e}")
            raise
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_code(
    prompt: str,
    target_dataset: str | None = None,
    llm_provider: str = "ollama",
    llm_model: str = "qwen2.5-coder:7b",
    df_dict: dict | None = None,
) -> str:
    """
    Generate Python transformation code using an LLM.
    
    Args:
        prompt: User's natural language transformation request
        target_dataset: Name of target dataset to transform
        llm_provider: "ollama" or "google"
        llm_model: Model name/ID
        df_dict: Dictionary of available dataframes (optional, for context)
    
    Returns:
        Python code with a transform(df_dict) function
    """
    # Extract column names if dataframes are provided
    columns = []
    sample_data = ""
    if df_dict and target_dataset and target_dataset in df_dict:
        df = df_dict[target_dataset]
        columns = list(df.columns)
        # Get first row as sample
        if df.__class__.__module__.startswith("pyspark.sql"):
            rows = df.limit(1).collect()
            if rows:
                sample_data = str(rows[0].asDict())
        else:
            if len(df) > 0:
                first_row = df.iloc[0].to_dict()
                sample_data = str(first_row)

    # Build the prompt with context
    full_prompt = build_code_generation_prompt(
        prompt,
        target_dataset or "unknown",
        columns,
        sample_data
    )

    # Create LLM instance
    llm = _create_llm_instance(llm_provider, llm_model)

    # Call LLM
    logger.info(f"Calling {llm_provider} LLM ({llm_model}) for code generation")
    response = llm.invoke(full_prompt)

    # Extract response text
    response_text = response if isinstance(response, str) else response.content

    # Extract code block from response
    code = _extract_code_block(response_text)

    if not code:
        logger.error("No code extracted from LLM response")
        raise ValueError("LLM did not generate valid code")

    logger.info(f"Code generated successfully ({len(code)} chars)")
    return code
