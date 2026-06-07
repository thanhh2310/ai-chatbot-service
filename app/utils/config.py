# File: app/utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    AI_CHAT_MODEL = os.getenv("AI_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    AI_ROUTER_MODEL = os.getenv("AI_ROUTER_MODEL", AI_CHAT_MODEL)
    AI_USE_LLM_INTENT = os.getenv("AI_USE_LLM_INTENT", "false").lower() in {"1", "true", "yes", "on"}
    AI_MAX_REPLY_TOKENS = int(os.getenv("AI_MAX_REPLY_TOKENS", "280"))
    AI_CACHE_TTL_SECONDS = int(os.getenv("AI_CACHE_TTL_SECONDS", "90"))
    PORT = int(os.getenv("FLASK_PORT", 5001))
