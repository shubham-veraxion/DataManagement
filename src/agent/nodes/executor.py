from src.core.logging.logger import get_logger

logger = get_logger(__name__)

def execute(code: str, df_dict: dict):
    local_env = {}

    try:
        exec(code, local_env, local_env)
        transform_fn = local_env.get("transform")
        if not callable(transform_fn):
            raise ValueError("Generated code must define a callable transform(df_dict) function")
        result = transform_fn(df_dict)

        logger.info("Execution success")
        return result

    except Exception:
        logger.exception("Execution failed")
        raise