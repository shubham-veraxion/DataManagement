from dataclasses import dataclass, field

@dataclass
class AgentState:
    prompt: str
    active_dataset: str = ""
    code: str = ""
    approved: bool = False
    snapshot_id: str = ""
    result_path: str = ""
    last_error: str = ""
    history: list = field(default_factory=list)