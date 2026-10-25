"""Thin Groq wrapper with retry on rate limits."""
import json
import re
import time

from . import config

_client = None


def client():
    global _client
    if _client is None:
        from groq import Groq
        key = config.secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (env, .env or Colab Secrets).")
        _client = Groq(api_key=key)
    return _client


def chat(messages, model=None, temperature=0.1, max_tokens=700, json_mode=False, retries=6):
    model = model or config.GROQ_MODEL
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
        kwargs["max_tokens"] = max_tokens + 800  # leave room for hidden reasoning
    delay = 4
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # rate limits / transient errors
            msg = str(e)
            if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                continue
            if attempt == retries - 1 or not re.search(r"429|rate|timeout|50\d|overloaded|json_validate", msg, re.I):
                raise
            wait = re.search(r"try again in ([\d.]+)s", msg)
            time.sleep(float(wait.group(1)) + 1 if wait else delay)
            delay = min(delay * 2, 60)


def chat_json(messages, model=None):
    raw = chat(messages, model=model or config.ROUTER_MODEL, temperature=0, max_tokens=300, json_mode=True)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    return json.loads(m.group(0) if m else raw)
