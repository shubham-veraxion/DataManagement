from src.agent.nodes.planner import plan
from src.agent.nodes.code_generator import generate_code
from src.agent.nodes.validator import validate
from src.agent.nodes.executor import execute
from src.core.snapshot.snapshot_manager import create_snapshot
from src.core.transformations.runner import save_output

def run_agent(state, df_dict, output_path):
    target_dataset = state.active_dataset or (next(iter(df_dict.keys())) if df_dict else "")
    target_df_dict = {target_dataset: df_dict[target_dataset]} if target_dataset in df_dict else df_dict

    # 1. Plan
    instruction = plan(state.prompt)

    # 2. Generate code only once until the prompt changes
    if not state.code:
        state.code = generate_code(instruction, target_dataset=target_dataset)

        # 3. Validate
        if not validate(state.code):
            raise Exception("Invalid code")

        # 4. Approval check
        if not state.approved:
            return None, state

    if not state.approved:
        return None, state

    # 5. Execute approved code against the selected dataset
    result_df = execute(state.code, target_df_dict)

    # 6. Save result
    save_output(result_df, output_path)
    state.result_path = output_path

    # 7. Snapshot
    state.snapshot_id = create_snapshot(
        output_path,
        state.code,
        {"prompt": state.prompt}
    )

    state.history.append(state.snapshot_id)
    state.last_error = ""

    return result_df, state