import sys
import tempfile
from pathlib import Path

import streamlit as st
from streamlit_ace import st_ace

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agent.graph import run_agent
from src.agent.state import AgentState
from src.core.ingestion.csv_loader import load_csv
from src.core.snapshot.version_store import (
    list_snapshots,
    load_snapshot_data,
)

DATA_DIR = Path("data")
OUTPUT_PATH = Path("storage/processed/temp.csv")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _is_spark_df(df) -> bool:
    return df.__class__.__module__.startswith("pyspark.sql")


def _preview_metrics(df):
    if _is_spark_df(df):
        preview_df = df.limit(20).toPandas()
        return preview_df, df.count(), len(df.columns)

    return df.head(20), len(df), len(df.columns)


def _schema_rows(df):
    if _is_spark_df(df):
        return [{"column": c, "type": t} for c, t in df.dtypes]

    return [{"column": c, "type": str(t)} for c, t in df.dtypes.items()]


def load_dataframes(uploaded_files) -> dict[str, object]:
    dataframes = {}

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            dataframes[Path(uploaded_file.name).stem] = load_csv(tmp_path)

        return dataframes

    if DATA_DIR.exists():
        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            dataframes[csv_path.stem] = load_csv(str(csv_path))

    return dataframes


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

DEFAULT_SESSION_STATE = {
    "agent_state": None,
    "dataframes": {},
    "active_dataset": "",
    "restored_prompt": "",
    "loaded_file_signature": "",
    "loaded_source": "",
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------------------------------------------------
# UI Config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic MDM Transformer",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');

    :root {
        --bg: #f8f2e9;
        --panel: #fff7ee;
        --ink: #1f1b16;
        --muted: #6f6257;
        --accent: #c76b2a;
        --line: #ead9c8;
    }

    section.main {
        background: linear-gradient(180deg, #fcf7f1 0%, #f2e7d9 100%);
    }

    .block-container {
        padding-top: 2rem;
    }

    .hero {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 40px rgba(41, 26, 15, 0.08);
    }

    .title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--ink);
    }

    .sub {
        color: var(--muted);
        margin-top: 0.4rem;
    }

    .label {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.7rem;
        color: var(--muted);
    }

    .chip {
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
    <div class="hero">
        <div class="label">Agentic MDM Studio</div>
        <div class="title">Transform, Join, and Snapshot at Scale</div>
        <div class="sub">
            Work across multiple datasets with PySpark-powered
            transformations and versioned snapshots.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Intake
# -----------------------------------------------------------------------------

st.markdown(" ")
st.markdown("<div class='label'>Data Intake</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

dataframes = load_dataframes(uploaded_files)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:

    st.subheader("⚙️ LLM Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "google"],
    )

    llm_model = st.text_input(
        "Model",
        value="qwen2.5-coder:7b"
        if llm_provider == "ollama"
        else "gemini-pro",
    )

    st.divider()

    st.subheader("📸 Snapshots")

    snapshots = list_snapshots()

    if snapshots:

        snapshot_options = [
            s["snapshot_id"] for s in snapshots
        ]

        selected_snapshot = st.selectbox(
            "Available snapshots",
            snapshot_options,
        )

        snapshot_info = next(
            s for s in snapshots
            if s["snapshot_id"] == selected_snapshot
        )

        snapshot_prompt = snapshot_info.get("prompt", "N/A")

        st.info(f"Prompt: {snapshot_prompt}")

        with st.expander("Details"):
            st.json(snapshot_info)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Load", use_container_width=True):

                if st.session_state.active_dataset:
                    snapshot_df = load_snapshot_data(selected_snapshot)

                    st.session_state.dataframes[
                        st.session_state.active_dataset
                    ] = snapshot_df

                    st.success(f"Loaded: {selected_snapshot}")
                    st.rerun()

        with col2:
            if st.button("🔄 Rollback", use_container_width=True):

                if st.session_state.active_dataset:

                    snapshot_df = load_snapshot_data(selected_snapshot)

                    st.session_state.dataframes[
                        st.session_state.active_dataset
                    ] = snapshot_df

                    st.session_state.agent_state = None
                    st.session_state.restored_prompt = snapshot_prompt

                    st.success(f"Rolled back: {selected_snapshot}")
                    st.rerun()

    else:
        st.info("No snapshots available.")


# -----------------------------------------------------------------------------
# Dataset State Handling
# -----------------------------------------------------------------------------

if not uploaded_files and st.session_state.dataframes:
    dataframes = st.session_state.dataframes

if not dataframes:
    st.info("📂 Upload CSV files or place CSVs in `data/` folder.")
    st.stop()

if uploaded_files:
    current_source = "uploaded"
    current_signature = "uploaded:" + ",".join(
        sorted(Path(f.name).stem for f in uploaded_files)
    )
else:
    current_source = "data_dir"
    current_signature = "data_dir:" + ",".join(
        sorted(dataframes.keys())
    )

if (
    current_signature != st.session_state.loaded_file_signature
    or not st.session_state.dataframes
):
    st.session_state.dataframes = dataframes
    st.session_state.loaded_file_signature = current_signature
    st.session_state.loaded_source = current_source


# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------

dataset_names = list(st.session_state.dataframes.keys())

default_index = 0

if st.session_state.active_dataset in dataset_names:
    default_index = dataset_names.index(
        st.session_state.active_dataset
    )

st.markdown("<div class='label'>Workspace</div>", unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    active_dataset = st.selectbox(
        "Select dataset",
        dataset_names,
        index=default_index,
    )

    st.session_state.active_dataset = active_dataset

with right:
    source_label = (
        "Uploads"
        if st.session_state.loaded_source == "uploaded"
        else "Data Folder"
    )

    st.markdown(
        f"""
        <div class="hero" style="padding: 1rem;">
            <div class="label">Source</div>
            <span class="chip">{source_label}</span>
            <span class="chip">{len(dataset_names)} datasets</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Data Preview
# -----------------------------------------------------------------------------

current_df = st.session_state.dataframes[active_dataset]

preview_df, row_count, col_count = _preview_metrics(current_df)

data_tab, code_tab = st.tabs([
    "Data Preview",
    "Code Editor",
])

with data_tab:

    preview_tab, schema_tab = st.tabs([
        "Preview",
        "Schema",
    ])

    with preview_tab:
        st.dataframe(preview_df, use_container_width=True)

    with schema_tab:
        st.dataframe(
            _schema_rows(current_df),
            use_container_width=True,
        )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", row_count)
    c2.metric("Columns", col_count)
    c3.metric("Dataset", active_dataset)


# -----------------------------------------------------------------------------
# Editable Code Editor
# -----------------------------------------------------------------------------

with code_tab:

    code = (
        st.session_state.agent_state.code
        if st.session_state.agent_state
        else ""
    )

    edited_code = st_ace(
        value=code,
        language="python",
        theme="monokai",
        height=600,
        font_size=14,
        key="code_editor",
        auto_update=True,
    )

    if st.session_state.agent_state:
        st.session_state.agent_state.code = edited_code

    if not code:
        st.info("Generate code to start editing.")


# -----------------------------------------------------------------------------
# Transformation Studio
# -----------------------------------------------------------------------------

st.markdown(" ")
st.subheader("Transformation Studio")

agent_state = st.session_state.agent_state

if agent_state and agent_state.log_path:
    st.caption(f"Run log: {agent_state.log_path}")

if agent_state and agent_state.last_error:
    st.error(agent_state.last_error)

prompt = st.text_area(
    "Describe the transformation",
    value=st.session_state.restored_prompt,
    height=160,
    placeholder=(
        "Example: join EOD with INT on market_area, "
        "filter null prices, aggregate hourly"
    ),
)

st.session_state.restored_prompt = ""

col1, col2 = st.columns(2)

generate_clicked = col1.button(
    "🔨 Generate Code",
    use_container_width=True,
)

approve_clicked = col2.button(
    "✅ Approve & Execute",
    use_container_width=True,
)


# -----------------------------------------------------------------------------
# Generate Code
# -----------------------------------------------------------------------------

if generate_clicked:

    state = AgentState(
        prompt=prompt,
        active_dataset=active_dataset,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    try:
        with st.spinner("Generating code..."):

            _, state = run_agent(
                state,
                st.session_state.dataframes,
                str(OUTPUT_PATH),
            )

        st.session_state.agent_state = state

        st.success("Code generated successfully.")

    except Exception as exc:

        state.last_error = str(exc)
        st.session_state.agent_state = state

        st.error(f"Error: {exc}")


# -----------------------------------------------------------------------------
# Execute Code
# -----------------------------------------------------------------------------

if (
    approve_clicked
    and st.session_state.agent_state
    and st.session_state.agent_state.code
):

    st.session_state.agent_state.approved = True

    try:

        with st.spinner("Executing transformation..."):

            result, state = run_agent(
                st.session_state.agent_state,
                st.session_state.dataframes,
                str(OUTPUT_PATH),
            )

        st.session_state.agent_state = state

        if result is not None:

            st.session_state.dataframes[
                active_dataset
            ] = result

            st.success(
                f"✨ Snapshot created: {state.snapshot_id}"
            )

            st.rerun()

        else:
            st.warning("No result returned.")

    except Exception as exc:

        st.session_state.agent_state.last_error = str(exc)

        st.error(f"Error: {exc}")