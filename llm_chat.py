# -*- coding: utf-8 -*-
"""LLM 调用：支持 SiliconFlow、阿里云百炼、OpenRouter（OpenAI 兼容接口）"""

import os
import time
import logging
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from types import SimpleNamespace
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY")
LOG_FILE = os.getenv("LOG_FILE", "ask.log")

# OpenAI 兼容接口的 base_url
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# QPS 限流（全局、线程安全）：上次请求时间与最小间隔
_rate_limit_lock = threading.Lock()
_rate_limit_last_time = 0.0


def _wait_rate_limit():
    """在发起 LLM 请求前调用，保证不超过配置的 QPS。"""
    global _rate_limit_last_time
    try:
        from config import LLM_MAX_QPS
    except ImportError:
        return
    if LLM_MAX_QPS <= 0:
        return
    min_interval = 1.0 / LLM_MAX_QPS
    with _rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _rate_limit_last_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _rate_limit_last_time = time.monotonic()

logger = logging.getLogger("LLMChatLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 关闭 httpx/httpcore 的每条请求 INFO 日志（openai 调阿里云等时使用），只保留 WARNING 及以上
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LLMChat:
    _session = None

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            cls._session = requests.Session()
            # 429 由下方 get_selicon_completion_once 内单独退避，此处只重试 5xx
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
            cls._session.mount("https://", adapter)
        return cls._session

    def __init__(self):
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.messages = [self.system_message]

    def dict_to_obj(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self.dict_to_obj(v) for k, v in d.items()})
        if isinstance(d, list):
            return [self.dict_to_obj(i) for i in d]
        return d

    def get_selicon_completion_once(self, question: str, model: str, enable_thinking: bool = False):
        if not (SILICONFLOW_API_KEY or "").strip():
            raise ValueError(
                "SILICONFLOW_API_KEY 未设置。请在运行容器时通过 --env-file 传入 .env，或设置环境变量 SILICONFLOW_API_KEY。"
            )
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "enable_thinking": enable_thinking,
        }
        session = self._get_session()
        try:
            backoff_sec = 60
            max_429_retries = 5
            try:
                from config import LLM_429_BACKOFF_SECONDS, LLM_429_MAX_RETRIES
                backoff_sec = LLM_429_BACKOFF_SECONDS
                max_429_retries = LLM_429_MAX_RETRIES
            except ImportError:
                pass
            for attempt in range(max_429_retries + 1):
                _wait_rate_limit()
                resp = session.post(
                    "https://api.siliconflow.cn/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=(10, 120),
                )
                if resp.status_code == 429:
                    wait_sec = backoff_sec
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            wait_sec = max(wait_sec, int(retry_after))
                        except ValueError:
                            pass
                    if attempt < max_429_retries:
                        logger.warning(
                            "SiliconFlow 429 限流，等待 %s 秒后重试 (第 %s/%s 次)",
                            wait_sec, attempt + 1, max_429_retries,
                        )
                        time.sleep(wait_sec)
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                raw = resp.json()
                return self.dict_to_obj(raw)
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout for model {model}: {e}")
            raise ValueError(f"SiliconFlow 请求超时 (model={model}): {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            # 便于排查：401=API Key 错误，4xx/5xx 或连接失败会带响应或原因
            detail = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = f"status={e.response.status_code} body={e.response.text[:200]}"
                except Exception:
                    pass
            raise ValueError(f"SiliconFlow 请求失败 (model={model}): {detail}") from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ValueError(f"SiliconFlow 调用异常 (model={model}): {e}") from e

    def _get_openai_completion_once(self, question: str, model: str, base_url: str, api_key: str):
        """通过 OpenAI 兼容接口请求（阿里云、OpenRouter 等）。"""
        if not (api_key or "").strip():
            raise ValueError("未设置对应平台的 API Key，请在 .env 中配置。")
        client = OpenAI(api_key=api_key.strip(), base_url=base_url, timeout=120.0)
        _wait_rate_limit()
        resp = client.chat.completions.create(
            model=model,
            messages=[self.system_message, {"role": "user", "content": question}],
            extra_body={"enable_thinking": False},
        )
        return self.dict_to_obj(resp.model_dump())

    def get_completion_once(self, question: str, model: str, mode: str = None, enable_thinking=False):
        if mode is None:
            try:
                from config import LLM_PLATFORM
                mode = LLM_PLATFORM
            except ImportError:
                mode = "siliconflow"
        mode = (mode or "siliconflow").lower().strip()
        if mode == "siliconflow":
            completion = self.get_selicon_completion_once(question, model, enable_thinking)
            return completion
        if mode == "aliyun":
            completion = self._get_openai_completion_once(
                question, model, ALIYUN_BASE_URL, DASHSCOPE_API_KEY or ""
            )
            return completion
        if mode == "openrouter":
            completion = self._get_openai_completion_once(
                question, model, OPENROUTER_BASE_URL, OPENROUTER_API_KEY or ""
            )
            return completion
        raise ValueError(f"Unsupported mode: {mode}，支持: siliconflow / aliyun / openrouter")

    def ask_once_with_usage(self, question: str, model: str, mode: str = None, enable_thinking: bool = False):
        try:
            completion = self.get_completion_once(question, model, mode, enable_thinking)
            if not hasattr(completion, "choices") or not completion.choices:
                raise ValueError("Invalid completion: no choices")
            msg = completion.choices[0].message
            if not hasattr(msg, "content"):
                raise ValueError("Invalid completion: no message content")
            response = msg.content
            usage = (
                completion.usage
                if hasattr(completion, "usage") and completion.usage
                else None
            )
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }
            return response, usage_dict
        except Exception as e:
            logger.error(f"Error in ask_once_with_usage: {e}")
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
