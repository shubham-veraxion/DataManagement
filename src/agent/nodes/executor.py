from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def execute(code: str, df_dict: dict):
    local_env = {}

    try:
        exec(code, local_env, local_env)
        result = local_env["transform"](df_dict)

        logger.info("Execution success")
        return result

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise