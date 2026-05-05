import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from src.agent.graph import run_agent
from src.agent.state import AgentState
from src.core.ingestion.csv_loader import load_csv
from src.core.snapshot.version_store import list_snapshots, load_snapshot_data


DATA_DIR = Path("data")
OUTPUT_PATH = Path("storage/processed/temp.csv")


def load_dataframes(uploaded_files) -> dict[str, pd.DataFrame]:
    dataframes: dict[str, pd.DataFrame] = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            dataframes[Path(uploaded_file.name).stem] = pd.read_csv(uploaded_file)
        return dataframes

    if DATA_DIR.exists():
        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            dataframes[csv_path.stem] = load_csv(str(csv_path))

    return dataframes


st.set_page_config(page_title="Agentic MDM Transformer", layout="wide")
st.title("Agentic MDM Transformer")
st.caption("Select a dataset, describe the transformation, approve the generated code, and persist a snapshot.")

# Sidebar for LLM configuration
with st.sidebar:
    st.subheader("LLM Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "google"],
        index=0,
        help="Select between local Ollama or Google Generative AI"
    )
    
    if llm_provider == "ollama":
        llm_model = st.text_input(
            "Ollama Model",
            value="qwen2.5-coder:7b",
            help="e.g., qwen2.5-coder:7b, qwen3:8b, gemma3:4b"
        )
    else:
        llm_model = st.text_input(
            "Google Model",
            value="gemini-pro",
            help="e.g., gemini-pro, gemini-1.5-pro"
        )

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
dataframes = load_dataframes(uploaded_files)

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = ""

if dataframes:
    st.session_state.dataframes = dataframes
    dataset_names = list(dataframes.keys())

    default_index = 0
    if st.session_state.active_dataset in dataset_names:
        default_index = dataset_names.index(st.session_state.active_dataset)
    elif any("EOD" in name.upper() for name in dataset_names):
        default_index = next(i for i, name in enumerate(dataset_names) if "EOD" in name.upper())

    active_dataset = st.selectbox("Active dataset", dataset_names, index=default_index)
    st.session_state.active_dataset = active_dataset

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Preview")
        st.dataframe(dataframes[active_dataset].head(20), use_container_width=True)
        st.write(f"Rows: {len(dataframes[active_dataset])} | Columns: {len(dataframes[active_dataset].columns)}")

    with right:
        prompt = st.text_area(
            "Transformation prompt",
            placeholder="Example: remove rows where price is null or zero",
            height=160,
        )

        if st.button("Generate Code", use_container_width=True):
            state = AgentState(
                prompt=prompt,
                active_dataset=active_dataset,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            try:
                result, state = run_agent(state, dataframes, str(OUTPUT_PATH))
                st.session_state.agent_state = state
                st.code(state.code or "")
                st.info("Code generated. Review it, then approve to execute.")
            except Exception as exc:
                state.last_error = str(exc)
                st.session_state.agent_state = state
                st.error(str(exc))

        if st.session_state.agent_state and st.session_state.agent_state.code:
            if st.button("Approve & Execute", use_container_width=True):
                st.session_state.agent_state.approved = True
                try:
                    result, state = run_agent(
                        st.session_state.agent_state,
                        dataframes,
                        str(OUTPUT_PATH),
                    )
                    st.session_state.agent_state = state
                    if result is not None:
                        st.success(f"Snapshot created: {state.snapshot_id}")
                        st.dataframe(result.head(20), use_container_width=True)
                    else:
                        st.warning("Execution did not return a result.")
                except Exception as exc:
                    st.session_state.agent_state.last_error = str(exc)
                    st.error(str(exc))

    st.divider()
    st.subheader("Snapshots")
    snapshots = list_snapshots()
    if snapshots:
        snapshot_options = [snapshot["snapshot_id"] for snapshot in snapshots]
        selected_snapshot = st.selectbox("Available snapshots", snapshot_options)
        selected_snapshot_info = next(item for item in snapshots if item["snapshot_id"] == selected_snapshot)
        st.json(selected_snapshot_info)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Snapshot", use_container_width=True):
                snapshot_df = load_snapshot_data(selected_snapshot)
                st.session_state.dataframes[active_dataset] = snapshot_df
                st.session_state.active_dataset = active_dataset
                st.success(f"Loaded snapshot {selected_snapshot} into {active_dataset}")
                st.dataframe(snapshot_df.head(20), use_container_width=True)
        
        with col2:
            if st.button("Rollback to Snapshot", use_container_width=True):
                snapshot_df = load_snapshot_data(selected_snapshot)
                st.session_state.dataframes[active_dataset] = snapshot_df
                st.session_state.active_dataset = active_dataset
                # Clear agent state to force fresh generation on snapshot data
                st.session_state.agent_state = None
                st.success(f"Rolled back to snapshot {selected_snapshot}. Data updated.")
                st.rerun()
    else:
        st.info("No snapshots available yet.")
else:
    st.info("Upload one or more CSV files, or place CSVs in the `data/` folder.")