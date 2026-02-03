# -*- coding: utf-8 -*-
"""
子系统 Worker 主程序：从中央服务器领取任务，提取诗歌地名，写回中央库并上报完成。
部署在子系统，循环执行：claim -> 拉诗 -> 提取地名 -> 写 place_names_match_results -> complete。
"""

import time
import logging

from config import (
    CENTRAL_API_BASE_URL,
    DEFAULT_MODEL,
    PROMPT_ID,
    MAX_WORKERS,
    TASK_TIMEOUT,
    MAX_CHARS_PER_BATCH,
    MAX_ITEMS_PER_BATCH,
    MAX_RETRIES,
    POLL_INTERVAL,
)
from task_client import claim_task, complete_task, health_check
from central_db import get_poems_by_ids, insert_match_results
from place_extractor import run_extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_one_task():
    """
    领取一个任务、拉诗、提取地名、写库、上报完成。
    返回: True 表示处理了一个任务，False 表示没有可领任务或出错。
    """
    ok, task_id, poem_ids, msg = claim_task()
    if not ok or task_id is None or poem_ids is None:
        logger.info("领取任务: %s", msg or "无待处理任务")
        return False

    logger.info("领取任务 task_id=%s, poem_ids 数量=%s", task_id, len(poem_ids))

    logger.info("正在从中央库拉取 %s 条诗歌...", len(poem_ids))
    poems = get_poems_by_ids(poem_ids)
    logger.info("已拉取 %s 条诗歌，开始地名提取", len(poems))
    if len(poems) != len(poem_ids):
        missing = set(poem_ids) - {p[0] for p in poems}
        logger.warning("中央库中部分诗歌缺失: 请求 %s 条，得到 %s 条，缺失 id: %s", len(poem_ids), len(poems), missing)

    if not poems:
        logger.warning("任务 %s 无有效诗歌，仍上报完成", task_id)
        ok2, msg2 = complete_task(task_id)
        logger.info("上报完成: success=%s, message=%s", ok2, msg2)
        return True

    try:
        results = run_extraction(
            poems,
            model=DEFAULT_MODEL,
            prompt_id=PROMPT_ID,
            max_workers=MAX_WORKERS,
            task_timeout=TASK_TIMEOUT,
            max_chars_per_batch=MAX_CHARS_PER_BATCH,
            max_items_per_batch=MAX_ITEMS_PER_BATCH,
            max_retries=MAX_RETRIES,
        )
    except Exception as e:
        logger.exception("地名提取失败 task_id=%s: %s", task_id, e)
        # 不写库、不上报完成，任务会一直 in_progress；也可选择写入失败记录后上报 failed，此处简单处理为不完成
        return True  # 避免死循环重试同一任务

    # 写入中央库 place_names_match_results
    insert_match_results(task_id, results)
    logger.info("任务 task_id=%s 已写入 %s 条结果", task_id, len(results))

    ok2, msg2 = complete_task(task_id)
    if not ok2:
        logger.error("上报完成失败 task_id=%s: %s", task_id, msg2)
    else:
        logger.info("任务 task_id=%s 已完成并上报", task_id)
    return True


def main():
    logger.info("Worker 启动，中央服务器: %s", CENTRAL_API_BASE_URL)
    ok, msg = health_check()
    if not ok:
        logger.warning("中央服务器健康检查失败: %s，将继续尝试领取任务", msg)
    else:
        logger.info("中央服务器健康检查: %s", msg)

    while True:
        try:
            processed = process_one_task()
            if not processed:
                logger.info("暂无任务，%s 秒后重试", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("收到中断，退出")
            break
        except Exception as e:
            logger.exception("单轮异常: %s", e)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
