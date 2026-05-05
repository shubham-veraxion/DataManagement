import ast
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

FORBIDDEN = ["os", "sys", "subprocess", "open"]

def validate(code: str) -> bool:
    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name in FORBIDDEN:
                        raise Exception(f"Forbidden import: {n.name}")

        logger.info("Validation passed")
        return True

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False