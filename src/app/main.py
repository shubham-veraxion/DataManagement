import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import tempfile

from src.agent.graph import run_agent
from src.agent.state import AgentState
from src.core.ingestion.csv_loader import load_csv
from src.core.snapshot.version_store import list_snapshots, load_snapshot_data


DATA_DIR = Path("data")
OUTPUT_PATH = Path("storage/processed/temp.csv")


def _is_spark_df(df) -> bool:
    return df.__class__.__module__.startswith("pyspark.sql")


def _preview_metrics(df):
    if _is_spark_df(df):
        preview_df = df.limit(20).toPandas()
        row_count = df.count()
        col_count = len(df.columns)
        return preview_df, row_count, col_count

    preview_df = df.head(20)
    return preview_df, len(df), len(df.columns)


def _schema_rows(df):
    if _is_spark_df(df):
        return [{"column": name, "type": dtype} for name, dtype in df.dtypes]

    return [{"column": name, "type": str(dtype)} for name, dtype in df.dtypes.items()]


def load_dataframes(uploaded_files) -> dict[str, object]:
    dataframes: dict[str, object] = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            dataframes[Path(uploaded_file.name).stem] = load_csv(str(tmp_path))
        return dataframes

    if DATA_DIR.exists():
        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            dataframes[csv_path.stem] = load_csv(str(csv_path))

    return dataframes


st.set_page_config(page_title="Agentic MDM Transformer", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
    :root {
        --mdm-bg: #f8f2e9;
        --mdm-panel: #fff7ee;
        --mdm-ink: #1f1b16;
        --mdm-muted: #6f6257;
        --mdm-accent: #c76b2a;
        --mdm-line: #ead9c8;
    }
    section.main {
        background: linear-gradient(180deg, #fcf7f1 0%, #f2e7d9 100%);
    }
    .block-container {
        padding-top: 2rem;
    }
    .mdm-hero {
        background: var(--mdm-panel);
        border: 1px solid var(--mdm-line);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 40px rgba(41, 26, 15, 0.08);
    }
    .mdm-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--mdm-ink);
        margin: 0;
    }
    .mdm-sub {
        color: var(--mdm-muted);
        margin-top: 0.4rem;
        font-size: 1rem;
    }
    .mdm-label {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.7rem;
        color: var(--mdm-muted);
    }
    .mdm-chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #ffe5cd;
        color: #7a3a0a;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mdm-hero">
        <div class="mdm-label">Agentic MDM Studio</div>
        <div class="mdm-title">Transform, Join, and Snapshot at Scale</div>
        <div class="mdm-sub">Work across multiple datasets with PySpark-powered transformations and versioned snapshots.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(" ")

st.markdown("<div class='mdm-label'>Data Intake</div>", unsafe_allow_html=True)
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
if "loaded_source" not in st.session_state:
    st.session_state.loaded_source = ""

if not uploaded_files and st.session_state.dataframes:
    dataframes = st.session_state.dataframes

if dataframes:
    if uploaded_files:
        current_source = "uploaded"
        current_file_signature = "uploaded:" + ",".join(
            sorted(Path(file.name).stem for file in uploaded_files)
        )
    else:
        current_source = "data_dir"
        current_file_signature = "data_dir:" + ",".join(sorted(dataframes.keys()))

    if (
        current_source == "data_dir"
        and st.session_state.loaded_source == "uploaded"
        and st.session_state.dataframes
    ):
        current_source = st.session_state.loaded_source
        current_file_signature = st.session_state.loaded_file_signature

    if current_file_signature != st.session_state.loaded_file_signature or not st.session_state.dataframes:
        st.session_state.dataframes = dataframes
        st.session_state.loaded_file_signature = current_file_signature
        st.session_state.loaded_source = current_source

    dataset_names = list(st.session_state.dataframes.keys())

    default_index = 0
    if st.session_state.active_dataset in dataset_names:
        default_index = dataset_names.index(st.session_state.active_dataset)
    elif any("EOD" in name.upper() for name in dataset_names):
        default_index = next(i for i, name in enumerate(dataset_names) if "EOD" in name.upper())

    st.markdown("<div class='mdm-label'>Workspace</div>", unsafe_allow_html=True)
    selector_left, selector_right = st.columns([2, 1])
    with selector_left:
        st.subheader("Active Dataset")
        active_dataset = st.selectbox("Select dataset", dataset_names, index=default_index, key="dataset_select")
        st.session_state.active_dataset = active_dataset

    with selector_right:
        source_label = "Uploads" if st.session_state.loaded_source == "uploaded" else "Data folder"
        st.markdown(
            f"""
            <div class="mdm-hero" style="padding: 1rem 1.2rem;">
                <div class="mdm-label">Source</div>
                <span class="mdm-chip">{source_label}</span>
                <span class="mdm-chip">{len(dataset_names)} datasets</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Use session_state dataframes for dynamic preview updates
    current_df = st.session_state.dataframes.get(active_dataset, dataframes.get(active_dataset))
    preview_df, row_count, col_count = _preview_metrics(current_df)

    st.markdown(" ")
    data_tab, code_tab = st.tabs(["Data Preview", "Code Preview"])
    with data_tab:
        preview_tab, schema_tab = st.tabs(["Preview", "Schema"])
        with preview_tab:
            st.dataframe(preview_df, use_container_width=True)
        with schema_tab:
            st.dataframe(_schema_rows(current_df), use_container_width=True)

        metric_left, metric_mid, metric_right = st.columns(3)
        with metric_left:
            st.metric("Rows", row_count, delta=None)
        with metric_mid:
            st.metric("Columns", col_count, delta=None)
        with metric_right:
            st.metric("Target", active_dataset, delta=None)
    with code_tab:
        code = st.session_state.agent_state.code if st.session_state.agent_state else ""
        if code:
            st.code(code, language="python")
        else:
            st.info("Generate code to preview it here.")

    st.markdown(" ")
    st.subheader("Transformation Studio")

    prompt_value = st.session_state.restored_prompt if st.session_state.restored_prompt else ""
    prompt = st.text_area(
        "Describe the transformation",
        placeholder="Example: join EOD with INT on market_area, filter null prices, and aggregate hourly",
        height=160,
        value=prompt_value,
        key="prompt_input"
    )

    if st.session_state.restored_prompt:
        st.session_state.restored_prompt = ""

    action_left, action_right = st.columns(2)
    with action_left:
        generate_clicked = st.button("🔨 Generate Code", use_container_width=True)
    with action_right:
        approve_clicked = st.button("✅ Approve & Execute", use_container_width=True)

    if generate_clicked:
        state = AgentState(
            prompt=prompt,
            active_dataset=active_dataset,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        try:
            result, state = run_agent(state, st.session_state.dataframes, str(OUTPUT_PATH))
            st.session_state.agent_state = state
            st.info("Review the generated code, then approve to execute.")
        except Exception as exc:
            state.last_error = str(exc)
            st.session_state.agent_state = state
            st.error(f"❌ Error: {str(exc)}")

    if approve_clicked and st.session_state.agent_state and st.session_state.agent_state.code:
        st.session_state.agent_state.approved = True
        try:
            result, state = run_agent(
                st.session_state.agent_state,
                st.session_state.dataframes,
                str(OUTPUT_PATH),
            )
            st.session_state.agent_state = state
            if result is not None:
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