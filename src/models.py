from langgchain_ollama import ChatOllama, OllamaEmbeddings
from decorator_utility import safe_execute

@safe_execute(default=None, log_error=True, reraise=False)
def get_chat_model(model_name, model_config):
    return ChatOllama(model=model_name, **model_config)

@safe_execute(default=None, log_error=True, reraise=False)
def get_embedding_model(model_name, model_config):
    return OllamaEmbeddings(model=model_name, **model_config)