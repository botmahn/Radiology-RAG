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
    def __init__(self, model: str, temperature: float = 0.2, top_p: float = 0.9, num_ctx: int = 4096):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx

    def ask(self, prompt: str, history: list[dict] | None = None, images: list[str] | None = None) -> str:
        """
        Send a chat turn with full history to Ollama.
        history: list of dicts like {"role": "system"|"user"|"assistant", "content": str}
        images: optional base64 strings (only used on the last user message)
        """
        # Build messages array from history (if provided)
        messages = []
        if history:
            for m in history:
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                if not content.strip():
                    continue
                messages.append({"role": role, "content": content})

        # Append current user turn
        user_msg = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        messages.append(user_msg)

        resp = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
            },
        )
        return resp["message"]["content"]