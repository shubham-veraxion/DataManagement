from src.agent.nodes.planner import plan
from src.agent.nodes.code_generator import generate_code
from src.agent.nodes.validator import validate
from src.agent.nodes.executor import execute
from src.core.snapshot.snapshot_manager import create_snapshot
from src.core.transformations.runner import save_output
from src.core.logging.logger import get_logger, start_new_run

logger = get_logger(__name__)

def run_agent(state, df_dict, output_path):
    target_dataset = state.active_dataset or (next(iter(df_dict.keys())) if df_dict else "")
    target_df_dict = {target_dataset: df_dict[target_dataset]} if target_dataset in df_dict else df_dict

    state.log_path = start_new_run()
    logger.info(
        "Agent run started | dataset=%s | provider=%s | model=%s",
        target_dataset,
        state.llm_provider,
        state.llm_model,
    )

    # 1. Plan
    try:
        instruction = plan(state.prompt)
    except Exception as exc:
        logger.exception("Planning failed")
        raise RuntimeError(f"Planning failed: {exc}") from exc

    # 2. Generate code only once until the prompt changes
    if not state.code:
        try:
            state.code = generate_code(
                instruction,
                target_dataset=target_dataset,
                llm_provider=state.llm_provider,
                llm_model=state.llm_model,
                df_dict=df_dict,
            )
        except Exception as exc:
            logger.exception("Code generation failed")
            raise RuntimeError(f"Code generation failed: {exc}") from exc

        # 3. Validate
        try:
            validate(state.code)
        except Exception as exc:
            logger.exception("Validation failed")
            raise RuntimeError(f"Validation failed: {exc}") from exc

        # 4. Approval check
        if not state.approved:
            logger.info("Awaiting approval before execution")
            return None, state

    if not state.approved:
        logger.info("Awaiting approval before execution")
        return None, state

    # Re-validate before execution (covers edits or older cached code)
    try:
        validate(state.code)
    except Exception as exc:
        logger.exception("Validation failed")
        raise RuntimeError(f"Validation failed: {exc}") from exc

    # 5. Execute approved code against the selected dataset
    try:
        result_df = execute(state.code, target_df_dict)
    except Exception as exc:
        logger.exception("Execution failed")
        raise RuntimeError(f"Execution failed: {exc}") from exc

    # 6. Save result
    try:
        save_output(result_df, output_path)
    except Exception as exc:
        logger.exception("Saving output failed")
        raise RuntimeError(f"Saving output failed: {exc}") from exc
    state.result_path = output_path

    # 7. Snapshot
    try:
        state.snapshot_id = create_snapshot(
            output_path,
            state.code,
            {"prompt": state.prompt}
        )
    except Exception as exc:
        logger.exception("Snapshot creation failed")
        raise RuntimeError(f"Snapshot creation failed: {exc}") from exc

    state.history.append(state.snapshot_id)
    state.last_error = ""

    return result_df, state