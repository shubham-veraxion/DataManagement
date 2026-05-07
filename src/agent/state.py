from dataclasses import dataclass, field

@dataclass
class AgentState:
    prompt: str
    active_dataset: str = ""
    code: str = ""
    approved: bool = False
    snapshot_id: str = ""
    result_path: str = ""
    log_path: str = ""
    last_error: str = ""
    history: list = field(default_factory=list)
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5-coder:7b"