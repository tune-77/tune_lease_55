import os
import sys
from components.ai_agent import run_agent_query

# Assume API key is available in environment or from .env
from ai_chat import _get_gemini_key_from_secrets, GEMINI_API_KEY_ENV
api_key = GEMINI_API_KEY_ENV or _get_gemini_key_from_secrets()

print("Testing agent run...")
if api_key:
    try:
        resp = run_agent_query("テスト", "テスト背景", api_key)
        print("Success:", resp[:100])
    except Exception as e:
        print("Error during run:", e)
else:
    print("No API key available to test, but import succeeded.")
