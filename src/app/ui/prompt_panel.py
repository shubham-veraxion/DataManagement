from __future__ import annotations

import streamlit as st


def render_prompt_header(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def render_dataset_choice(dataset_names: list[str], default_index: int = 0) -> str:
    if not dataset_names:
        return ""

    return st.selectbox("Active dataset", dataset_names, index=default_index)