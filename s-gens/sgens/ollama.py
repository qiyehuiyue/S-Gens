from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    timeout_seconds: int = 120


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)

    def ping(self) -> bool:
        try:
            req = request.Request(f"{self.config.base_url}/api/tags", method="GET")
            with request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str | None = None, temperature: float | None = None) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else self.config.temperature},
        }
        if system:
            payload["system"] = system
        try:
            response = self._post("/api/generate", payload)
        except error.URLError as exc:  # pragma: no cover - network/service dependent
            raise RuntimeError(
                f"Failed to reach Ollama at {self.config.base_url}. Make sure Ollama is installed and running."
            ) from exc
        return response.get("response", "").strip()
