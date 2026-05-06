import ast
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

FORBIDDEN = ["os", "sys", "subprocess", "open", "pandas"]

def validate(code: str) -> bool:
    tree = ast.parse(code)
    has_functions_import = False
    uses_F = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name in FORBIDDEN:
                    raise Exception(f"Forbidden import: {n.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module in FORBIDDEN:
                raise Exception(f"Forbidden import: {node.module}")
            if node.module == "pyspark.sql":
                for n in node.names:
                    if n.name == "functions" and n.asname == "F":
                        has_functions_import = True

        if isinstance(node, ast.Name) and node.id == "F" and isinstance(node.ctx, ast.Load):
            uses_F = True

    if uses_F and not has_functions_import:
        raise Exception("Missing required import: from pyspark.sql import functions as F")

    logger.info("Validation passed")
    return True