"""Default upstream base URLs and shared provider constants.

Adapters and :mod:`providers.registry` import from here to avoid duplicating
literals and to keep ``providers.registry`` free of per-adapter eager imports.
"""

# OpenAI-compatible chat (NIM, DeepSeek) and local/native provider endpoints
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
DEEPSEEK_DEFAULT_BASE = "https://api.deepseek.com"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
LLAMACPP_DEFAULT_BASE = "http://localhost:8080/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"
