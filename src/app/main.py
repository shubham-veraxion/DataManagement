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

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
dataframes = load_dataframes(uploaded_files)

# Sidebar for LLM configuration and snapshot management
with st.sidebar:
    st.subheader("⚙️ LLM Configuration")
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
    
    st.divider()
    
    # Snapshot management in sidebar
    st.subheader("📸 Snapshots")
    snapshots = list_snapshots()
    if snapshots:
        snapshot_options = [snapshot["snapshot_id"] for snapshot in snapshots]
        selected_snapshot = st.selectbox("Available snapshots", snapshot_options, key="snapshot_select")
        selected_snapshot_info = next(item for item in snapshots if item["snapshot_id"] == selected_snapshot)
        
        # Extract and display prompt from snapshot
        snapshot_prompt = selected_snapshot_info.get("prompt", "N/A") if isinstance(selected_snapshot_info, dict) else "N/A"
        st.info(f"**Prompt:** {snapshot_prompt}")
        
        with st.expander("📋 Full Details", expanded=False):
            st.json(selected_snapshot_info)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load", use_container_width=True, key="load_snapshot_btn"):
                if dataframes and "active_dataset" in st.session_state and st.session_state.active_dataset:
                    snapshot_df = load_snapshot_data(selected_snapshot)
                    st.session_state.dataframes[st.session_state.active_dataset] = snapshot_df
                    st.success(f"Loaded: {selected_snapshot}")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Rollback", use_container_width=True, key="rollback_snapshot_btn"):
                if dataframes and "active_dataset" in st.session_state and st.session_state.active_dataset:
                    snapshot_df = load_snapshot_data(selected_snapshot)
                    st.session_state.dataframes[st.session_state.active_dataset] = snapshot_df
                    st.session_state.agent_state = None
                    # Restore prompt from snapshot
                    st.session_state.restored_prompt = snapshot_prompt
                    st.success(f"Rolled back to: {selected_snapshot}")
                    st.rerun()
    else:
        st.info("No snapshots yet. Execute a transformation to create one.")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = ""
if "restored_prompt" not in st.session_state:
    st.session_state.restored_prompt = ""
if "loaded_file_signature" not in st.session_state:
    st.session_state.loaded_file_signature = ""

if dataframes:
    current_file_signature = (
        "uploaded:" + ",".join(sorted(Path(file.name).stem for file in uploaded_files))
        if uploaded_files
        else "data_dir:" + ",".join(sorted(dataframes.keys()))
    )

    if current_file_signature != st.session_state.loaded_file_signature or not st.session_state.dataframes:
        st.session_state.dataframes = dataframes
        st.session_state.loaded_file_signature = current_file_signature

    dataset_names = list(st.session_state.dataframes.keys())

    default_index = 0
    if st.session_state.active_dataset in dataset_names:
        default_index = dataset_names.index(st.session_state.active_dataset)
    elif any("EOD" in name.upper() for name in dataset_names):
        default_index = next(i for i, name in enumerate(dataset_names) if "EOD" in name.upper())

    st.subheader("📊 Active Dataset")
    active_dataset = st.selectbox("Select dataset", dataset_names, index=default_index, key="dataset_select")
    st.session_state.active_dataset = active_dataset

    # Use session_state dataframes for dynamic preview updates
    current_df = st.session_state.dataframes.get(active_dataset, dataframes.get(active_dataset))
    
    left, right = st.columns([1, 1])
    with left:
        st.subheader("🔍 Preview")
        st.dataframe(current_df.head(20), use_container_width=True)
        st.metric("Rows", len(current_df), delta=None)
        st.metric("Columns", len(current_df.columns), delta=None)

    with right:
        st.subheader("✏️ Transformation")
        
        # Use restored_prompt if available (from rollback), otherwise empty
        prompt_value = st.session_state.restored_prompt if st.session_state.restored_prompt else ""
        
        prompt = st.text_area(
            "Describe the transformation",
            placeholder="Example: remove rows where price is null or zero",
            height=140,
            value=prompt_value,
            key="prompt_input"
        )
        
        # Clear restored prompt after it's been displayed
        if st.session_state.restored_prompt:
            st.session_state.restored_prompt = ""

        if st.button("🔨 Generate Code", use_container_width=True):
            state = AgentState(
                prompt=prompt,
                active_dataset=active_dataset,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            try:
                result, state = run_agent(state, st.session_state.dataframes, str(OUTPUT_PATH))
                st.session_state.agent_state = state
                st.code(state.code or "", language="python")
                st.info("📋 Review the code. Click 'Approve & Execute' to run it.")
            except Exception as exc:
                state.last_error = str(exc)
                st.session_state.agent_state = state
                st.error(f"❌ Error: {str(exc)}")

        if st.session_state.agent_state and st.session_state.agent_state.code:
            if st.button("✅ Approve & Execute", use_container_width=True):
                st.session_state.agent_state.approved = True
                try:
                    result, state = run_agent(
                        st.session_state.agent_state,
                        st.session_state.dataframes,
                        str(OUTPUT_PATH),
                    )
                    st.session_state.agent_state = state
                    if result is not None:
                        # Update session state dataframes with the result
                        st.session_state.dataframes[active_dataset] = result
                        st.success(f"✨ Snapshot created: {state.snapshot_id}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Execution did not return a result.")
                except Exception as exc:
                    st.session_state.agent_state.last_error = str(exc)
                    st.error(f"❌ Error: {str(exc)}")
else:
    st.info("📂 Upload one or more CSV files, or place CSVs in the `data/` folder.")