# -*- coding: utf-8 -*-
"""中央 MySQL：按 poem_ids 拉取诗歌、写入 place_names_match_results"""

import pymysql
from pymysql.cursors import DictCursor

from config import MYSQL_CONFIG, POEM_TABLE


def get_connection(use_dict_cursor=True):
    kwargs = dict(MYSQL_CONFIG)
    if use_dict_cursor:
        kwargs["cursorclass"] = DictCursor
    return pymysql.connect(**kwargs)


def get_poems_by_ids(poem_ids):
    """
    根据 poem_ids 从中央库 quiz_poem_2 拉取诗歌。
    返回: [(id, title, dynasty, author, content_original), ...]，顺序与 poem_ids 一致（缺失的 id 不出现）。
    """
    if not poem_ids:
        return []
    conn = get_connection()
    try:
        placeholders = ",".join(["%s"] * len(poem_ids))
        sql = (
            f"SELECT id, title, dynasty, author, content_original "
            f"FROM `{POEM_TABLE}` WHERE id IN ({placeholders}) ORDER BY id"
        )
        with conn.cursor() as cur:
            cur.execute(sql, poem_ids)
            rows = cur.fetchall()
        # 保持与 poem_test 一致的元组格式 (id, title, dynasty, author, content_original)
        if rows and isinstance(rows[0], dict):
            return [
                (r["id"], r["title"] or "", r["dynasty"] or "", r["author"] or "", r["content_original"] or "")
                for r in rows
            ]
        return list(rows)
    finally:
        conn.close()


def insert_match_results(task_id: int, results: list):
    """
    将地名提取结果写入中央库 place_names_match_results。
    results: [(quiz_poem2_id: int, match_names: str), ...]
    """
    if not results:
        return
    conn = get_connection(use_dict_cursor=False)
    try:
        with conn.cursor() as cur:
            for poem_id, match_names in results:
                cur.execute(
                    "INSERT INTO place_names_match_results (quiz_poem2_id, match_names, task_id) VALUES (%s, %s, %s)",
                    (poem_id, match_names, task_id),
                )
        conn.commit()
    finally:
        conn.close()
