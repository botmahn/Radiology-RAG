import ollama
import yaml

def load_config(path: str = "configs/reportgen_config.yaml") -> dict:
    with open("configs/reportgen_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # sensible fallbacks
    cfg.setdefault("models", {})
    cfg.setdefault("ollama", {})
    cfg.setdefault("rag", {})
    cfg.setdefault("paths", {})
    return cfg

class OllamaChatSession:
    """
    Minimal chat wrapper over ollama.chat that keeps conversation context in memory.
    Use ONLY for post-report general chat (no RAG).
    """
    def __init__(self, model: str, temperature: float = 0.2, top_p: float = 0.9, num_ctx: int = 4096):
        self.model = model
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        }
        self.history = []  # [{'role': 'user'/'assistant', 'content': '...'}]

    def ask(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})
        resp = ollama.chat(
            model=self.model,
            options=self.params,
            messages=self.history,
        )
        reply = resp["message"]["content"].strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply