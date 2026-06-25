"""
Application configuration using Pydantic Settings.
All values can be overridden via environment variables or .env file.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- LLM Provider API Keys ---
    openai_api_key: str = Field(default="sk-fake-key", description="OpenAI API key")
    anthropic_api_key: str = Field(default="sk-ant-fake", description="Anthropic API key")
    groq_api_key: str = Field(default="gsk_fake", description="Groq API key")

    # --- LiteLLM Proxy ---
    litellm_proxy_url: str = "http://localhost:4000"
    litellm_master_key: str = "sk-litellm-master-key-1234"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_host: str = "localhost"
    redis_port: int = 6379

    # --- Langfuse ---
    langfuse_secret_key: str = Field(default="sk-lf-agentic-local-secret", description="Langfuse secret key")
    langfuse_public_key: str = Field(default="pk-lf-agentic-local-public", description="Langfuse public key")
    langfuse_host: str = "http://localhost:3001"

    # --- OpenTelemetry ---
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "agentic-ai-app"

    # --- Cache Settings ---
    cache_ttl_seconds: int = 3600          # 1 hour default TTL
    semantic_cache_threshold: float = 0.85  # Cosine similarity threshold

    # --- Agent Settings ---
    research_agent_model: str = "gpt-4o-mini"   # Via LiteLLM proxy
    synthesis_agent_model: str = "gpt-4o-mini"  # Via LiteLLM proxy
    max_agent_iterations: int = 5
    agent_timeout_seconds: int = 60

    # --- Guardrails ---
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True
    pii_detection_enabled: bool = True
    toxicity_threshold: float = 0.8

    # --- Loki (direct push — no Promtail needed) ---
    loki_url: str = "http://localhost:3100"

    # --- RAGAS ---
    # RAGAS runs as an offline job (scripts/run_ragas_eval.py), not inline.
    # Enable the /api/v1/eval/ragas endpoint in non-production envs only.
    ragas_enabled_in_api: bool = True  # set False in production


@lru_cache
def get_settings() -> Settings:
    return Settings()
