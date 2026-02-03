# -*- coding: utf-8 -*-
"""LLM 调用（复用 poem_test 逻辑，SiliconFlow 等）"""

import os
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
LOG_FILE = os.getenv("LOG_FILE", "ask.log")

logger = logging.getLogger("LLMChatLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class LLMChat:
    _session = None

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            cls._session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
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
            resp = session.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=(10, 120),
            )
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

    def get_completion_once(self, question: str, model: str, mode: str = "siliconflow", enable_thinking=False):
        if mode == "siliconflow":
            completion = self.get_selicon_completion_once(question, model, enable_thinking)
            return completion
        raise ValueError(f"Unsupported mode: {mode}")

    def ask_once_with_usage(self, question: str, model: str, mode: str = "siliconflow", enable_thinking: bool = False):
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
