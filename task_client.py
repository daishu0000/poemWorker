# -*- coding: utf-8 -*-
"""中央服务器任务 API 客户端：领取任务、上报完成"""

import requests
from config import CENTRAL_API_BASE_URL


def claim_task():
    """
    从中央服务器领取一个待处理任务。
    返回: (success: bool, task_id: int | None, poem_ids: list[int] | None, message: str)
    """
    url = f"{CENTRAL_API_BASE_URL.rstrip('/')}/api/task/claim"
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if not data.get("success"):
            return False, None, None, data.get("message", "领取失败")
        return True, data["task_id"], data["poem_ids"], ""
    except requests.RequestException as e:
        return False, None, None, str(e)
    except (KeyError, TypeError) as e:
        return False, None, None, f"响应格式异常: {e}"


def complete_task(task_id: int):
    """
    向中央服务器上报任务完成。
    返回: (success: bool, message: str)
    """
    url = f"{CENTRAL_API_BASE_URL.rstrip('/')}/api/task/complete"
    try:
        resp = requests.post(url, json={"task_id": task_id}, timeout=30)
        data = resp.json()
        if not data.get("success"):
            return False, data.get("message", "上报失败")
        return True, data.get("message", "ok")
    except requests.RequestException as e:
        return False, str(e)
    except (KeyError, TypeError) as e:
        return False, f"响应格式异常: {e}"


def health_check():
    """健康检查中央服务器。"""
    url = f"{CENTRAL_API_BASE_URL.rstrip('/')}/api/health"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return data.get("success", False), data.get("message", "")
    except Exception as e:
        return False, str(e)
