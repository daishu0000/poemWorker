# -*- coding: utf-8 -*-
"""地名提取逻辑（复用 poem_test get_place_name）：批量 LLM 分析诗歌，返回 (id, match_names)"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

# 进度输出间隔（秒）
PROGRESS_INTERVAL = 10

from llm_chat import LLMChat

try:
    from config import LLM_PLATFORM
    MODE = LLM_PLATFORM
except ImportError:
    MODE = "siliconflow"
ENABLE_THINKING = False


def _strip_code_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return raw


def parse_ai_response_simple(response: str, prompt_id: int = 1):
    if prompt_id == 2 or prompt_id == 3:
        raw = (response or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                raw = "\n".join(lines[1:-1]).strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "error" in obj:
                return 0, response
            has_place = obj.get("has_place", 0)
            try:
                has_place = 1 if int(has_place) else 0
            except Exception:
                has_place = 0
            compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            return has_place, compact
        except Exception:
            return 0, '{"error":"format_error"}'
    if prompt_id == 4:
        if response == ",":
            is_geo = 0
        else:
            is_geo = 1
        match_names = response
        return is_geo, match_names
    try:
        parts = response.split(",")
        is_geo = int(parts[0])
    except Exception:
        return 0, ""
    if is_geo:
        match_names = ",".join(parts[1:]) if len(parts) > 1 else ""
    else:
        match_names = ""
    return is_geo, match_names


def parse_ai_batch_response(response: str, expected_ids: Iterable[int], prompt_id: int) -> Dict[int, str]:
    expected_set = set(int(x) for x in expected_ids)
    raw = _strip_code_fence(response)
    out: Dict[int, str] = {}
    if prompt_id == 3:
        try:
            obj = json.loads(raw)
        except Exception:
            return {}
        if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
            obj = obj["results"]
        if not isinstance(obj, list):
            return {}
        for item in obj:
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                pid = int(item["id"])
            except Exception:
                continue
            if pid not in expected_set:
                continue
            has_place = item.get("has_place", 0)
            try:
                has_place = 1 if int(has_place) else 0
            except Exception:
                has_place = 0
            places = item.get("places", [])
            if not isinstance(places, list):
                places = []
            normalized = {"has_place": has_place, "places": places}
            out[pid] = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if prompt_id == 4:
        try:
            items = raw.split(";")
        except Exception:
            return {}
        for idx, item in enumerate(items):
            pid = list(expected_set)[idx]
            try:
                parts = item.split(",")
                is_geo = int(parts[0])
                compact = item.split(",", 1)[1] if len(parts) > 1 else ""
                if is_geo:
                    out[pid] = compact
                else:
                    out[pid] = ","
            except Exception:
                out[pid] = ","
    return out


def get_prompt(prompt_id: int = 1):
    prompt1 = """请利用搜索工具分析这首诗的题目和内容中是否包含可定位的地点信息(可以确定大致经纬度或大致范围),
    是返回1,地名(现代标准地名)其中现代标准地名依据地名尺度可选格式省-市-县-地名、省-市-地名、
    省-地名、地名，如果直接指行政区就用省-市-县即可，否返回0。如:\n
    1,西湖(浙江省-杭州市-西湖区-西湖),黄鹤楼(湖北省-武汉市-武昌区-黄鹤楼)
    0
    1,武昌(湖北省-武汉市),长安(陕西省-西安市),黄山(安徽省-黄山市-黄山区-黄山)
    请仔细观察是否有地名，不要看个大概。你只需要返回结果，不要返回其他多余的信息，
    也不需要返回分析过程。另外，如果你发现一个古地名现代地名未知，请不要输出这个古地名，
    请确保每个古地名和现代地名是能一一对应的，
    不要输出类似于小胡村(未知-小胡村),武穆坟(浙江省-杭州市) 这样不对应的信息。诗如下：\n\n"""

    prompt2 = """请分析这首诗的题目和内容中是否包含可定位的地点信息，注意地点需要是实指的点状地名，不能是虚指或超大范围地名(如山脉、较长河流、省份、大区等)
    请返回一个json,格式是
    {
        "has_place": 0或1,
        "places": [
        {
            "name": "地名",
            "modern_name": "现代标准地名",
            "country": "国家",
            "province": "省",
            "city": "市",
            "county": "县",
        }
        },
        ...
    }
    注意，如果某一项信息未知，请用null表示。另外，所有的行政区请加通名，如xx省，xx市，xx县
    诗如下:\n
    """

    prompt3 = """你将收到一个JSON数组poems，包含多首诗歌
请逐首判断题目和内容中是否包含可定位的地点信息。注意地点需要是实指的点状地名或小范围线/面状地名（如城市、山峰、桥梁、湖泊），不能是虚指或超大范围地名(如山脉、较长河流、省份、大区等)。
请仔细观察是否有地名，不要看个大概。你只需要返回结果，不要返回其他多余的信息，
也不需要返回分析过程。另外，如果你发现一个古地名现代地名未知，请不要输出这个古地名，
请严格返回一个JSON数组（不要输出任何解释、不要输出markdown代码块），数组中每个元素的格式为：
{"id":<诗歌id>,"has_place":0或1,"places":[{"name":"地名","modern_name":"现代标准地名","country":"国家或null","province":"省或null","city":"市或null","county":"县或null"}]}
要求：
1) 必须覆盖输入poems中的每一个id，且id必须与输入一致
2) 若has_place为0，则places必须是空数组[]
3) 若某一项信息未知，请用null表示，不要用\"未知\"、\"不确定\"等文字
4) 只返回JSON数组本身，保证可被json.loads直接解析
5) 地址请务必加上通名，如xx省，xx市，xx县
6) modern_name务必写现代地名，不要写古称
7) modern_name不要加上上级地址，如直接写“西湖”，不要写“浙江省-杭州市-西湖区-西湖”，如果是指行政区，则写name对应的级别即可
8) 请不要把城市古称对应到一些现代的同名区，如钱塘指的是杭州市而不是杭州市钱塘区，除非你能确定某个区确实市城市古称所指范围
9) 地址能精确就精确（尤其是点状地名，应尽可能精确到县级，面状地名应精确到最小包含的行政区）
10) 对于直辖市，写在province里，city留空，下面的区写在county里；对于县级市，写在county里，city写上对应的地级市

示例:
{"id":1,"has_place":1,"places":[{"name":"西湖","modern_name":"西湖","country":"中国","province":"浙江省","city":"杭州市","county":"西湖区"},{"name":"金陵","modern_name":"南京市","country":"中国","province":"江苏省","city":"南京市","county":null}]}

输入如下（JSON）：\n"""

    prompt4 = """请利用搜索工具分析以下若干首诗的题目和内容中是否包含可定位的地点信息(可以确定大致经纬度或大致范围),
    是返回1,地名(现代标准地名)其中现代标准地名依据地名尺度可选格式省-市-县-地名、省-市-地名、
    省-地名、地名，如果直接指行政区就用省-市-县即可，否返回0。不同诗歌结果按照给出的顺序排列并且用;分割,如:\n
    1,西湖(浙江省-杭州市-西湖区-西湖),黄鹤楼(湖北省-武汉市-武昌区-黄鹤楼);0;1,武昌(湖北省-武汉市),长安(陕西省-西安市),黄山(安徽省-黄山市-黄山区-黄山)
    请仔细观察是否有地名，不要看个大概。你只需要返回结果，不要返回其他多余的信息，
    也不需要返回分析过程。另外，如果你发现一个古地名现代地名未知，请不要输出这个古地名，
    请确保每个古地名和现代地名是能一一对应的，
    不要输出类似于小胡村(未知-小胡村),武穆坟(浙江省-杭州市) 这样不对应的信息。诗如下：\n\n"""

    prompt_dict = {1: prompt1, 2: prompt2, 3: prompt3, 4: prompt4}
    return prompt_dict.get(prompt_id, prompt3)


def _poem_to_obj(poem: Tuple[Any, ...]) -> Dict[str, Any]:
    pid = int(poem[0])
    title = str(poem[1]) if len(poem) > 1 else ""
    dynasty = str(poem[2]) if len(poem) > 2 else ""
    author = str(poem[3]) if len(poem) > 3 else ""
    content = str(poem[4]) if len(poem) > 4 else ""
    return {"id": pid, "content": f"{title} {dynasty} {author} {content}"}


def chunk_poems_by_chars(
    poems: List[Tuple[Any, ...]], max_chars: int = 6000, max_items: int = 12
) -> List[List[Tuple[Any, ...]]]:
    batches: List[List[Tuple[Any, ...]]] = []
    cur: List[Tuple[Any, ...]] = []
    cur_chars = 0
    for p in poems:
        obj = _poem_to_obj(p)
        size = len(json.dumps(obj, ensure_ascii=False))
        if size >= max_chars:
            if cur:
                batches.append(cur)
                cur = []
                cur_chars = 0
            batches.append([p])
            continue
        if (cur and (cur_chars + size > max_chars)) or (len(cur) >= max_items):
            batches.append(cur)
            cur = []
            cur_chars = 0
        cur.append(p)
        cur_chars += size
    if cur:
        batches.append(cur)
    return batches


def analyze_poems_batch_request(
    poems_batch: List[Tuple[Any, ...]], prompt: str, prompt_id: int, model: str
) -> Tuple[Dict[int, str], dict]:
    llm_chat = LLMChat()
    payload = {"poems": [_poem_to_obj(p) for p in poems_batch]}
    question = prompt + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        resp, usage_info = llm_chat.ask_once_with_usage(question, model, MODE, ENABLE_THINKING)
    except Exception as e:
        print(f"批量请求异常: {e}")
        return {}, usage_info
    expected_ids = [int(p[0]) for p in poems_batch]
    result = parse_ai_batch_response(resp, expected_ids, prompt_id)
    if not result and resp:
        preview = resp[:200] if len(resp) > 200 else resp
        print(f"批量解析失败，响应预览: {preview}...")
    return result, usage_info


def _progress_reporter(
    total: int,
    id_to_result: Dict[int, str],
    stop_event: threading.Event,
    interval: int = PROGRESS_INTERVAL,
) -> None:
    """后台线程：每 interval 秒输出一次地名提取进度，直到 stop_event 被设置。"""
    while True:
        if stop_event.wait(interval):
            break
        done = len(id_to_result)
        pct = (100 * done // total) if total else 0
        logger.info("地名提取进度: 已处理 %s / 共 %s 条 (%s%%)", done, total, pct)


def analyze_poems_batches_concurrent(
    poems: List[Tuple[Any, ...]],
    prompt: str,
    prompt_id: int,
    model: str,
    max_workers: int = 8,
    task_timeout: int | None = None,
    max_chars_per_batch: int = 1000,
    max_items_per_batch: int = 20,
    max_retries: int = 2,
) -> List[Tuple[int, str]]:
    """
    批量地名提取主入口。返回 [(poem_id, match_names_str), ...]，match_names_str 为 JSON 或 ',' 等。
    """
    batches = chunk_poems_by_chars(poems, max_chars=max_chars_per_batch, max_items=max_items_per_batch)
    id_to_result: Dict[int, str] = {}
    batch_to_ids = {tuple(b): [int(p[0]) for p in b] for b in batches}
    failed_batches = list(batches)

    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_progress_reporter,
        args=(len(poems), id_to_result, stop_event),
        kwargs={"interval": PROGRESS_INTERVAL},
        daemon=True,
    )
    progress_thread.start()

    for retry_count in range(max_retries + 1):
        if not failed_batches:
            break
        if retry_count > 0:
            print(f"\n开始第 {retry_count} 次重试，失败批次数量: {len(failed_batches)}")
        current_failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(analyze_poems_batch_request, b, prompt, prompt_id, model): b
                for b in failed_batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                ids = batch_to_ids[tuple(batch)]
                try:
                    batch_map, _ = future.result(timeout=task_timeout) if task_timeout else future.result()
                except Exception as e:
                    batch_map = {}
                    if retry_count == 0:
                        print(f"批次异常: {ids[0]}..{ids[-1]} - {e}")
                returned_ids = set(batch_map.keys())
                expected_ids = set(ids)
                missing_ids = expected_ids - returned_ids
                if missing_ids:
                    current_failed.append(batch)
                for pid, compact in batch_map.items():
                    id_to_result[pid] = compact
        failed_batches = current_failed

    stop_event.set()
    done = len(id_to_result)
    total = len(poems)
    pct = (100 * done // total) if total else 0
    logger.info("地名提取进度: 已完成 %s / 共 %s 条 (%s%%)", done, total, pct)
    return [(int(p[0]), id_to_result.get(int(p[0]), '{"error":"format_error"}')) for p in poems]


def run_extraction(
    poems: List[Tuple[Any, ...]],
    model: str,
    prompt_id: int = 3,
    max_workers: int = 8,
    task_timeout: int | None = 120,
    max_chars_per_batch: int = 1000,
    max_items_per_batch: int = 12,
    max_retries: int = 2,
) -> List[Tuple[int, str]]:
    """
    对诗歌列表做地名提取，返回 [(poem_id, match_names_str), ...]。
    match_names_str 为 prompt_id=3 时的 JSON 字符串，或 ',' 表示无地名。
    写入 place_names_match_results 时直接使用该字符串作为 match_names。
    """
    prompt = get_prompt(prompt_id)
    if prompt_id in (3, 4):
        raw_results = analyze_poems_batches_concurrent(
            poems,
            prompt,
            prompt_id,
            model,
            max_workers=max_workers,
            task_timeout=task_timeout,
            max_chars_per_batch=max_chars_per_batch,
            max_items_per_batch=max_items_per_batch,
            max_retries=max_retries,
        )
        return raw_results
    # 单首模式暂不在此实现，worker 仅用批量模式
    raise ValueError("Worker 仅支持 prompt_id 3 或 4 的批量模式")
