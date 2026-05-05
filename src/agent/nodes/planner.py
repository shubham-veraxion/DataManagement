from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def plan(prompt: str) -> str:
    """
    Convert user prompt into structured instruction.
    Keep simple for now.
    """
    logger.info("Planning step executed")
    return prompt