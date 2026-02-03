# -*- coding: utf-8 -*-
"""子系统 Worker 配置：中央服务器 API 与中央 MySQL"""

import os
from dotenv import load_dotenv

load_dotenv()

# 中央服务器 API（任务领取与完成）
CENTRAL_API_BASE_URL = os.getenv("CENTRAL_API_BASE_URL", "http://121.40.230.141:5001")

# 中央 MySQL（与 task_manager 同一库：读诗歌、写 place_names_match_results）
# connect_timeout/read_timeout 避免网络不可达时无限挂起
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "121.40.230.141"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "poem"),
    "password": os.getenv("MYSQL_PASSWORD", "TB!#8p+_Cp"),
    "database": os.getenv("MYSQL_DATABASE", "poem"),
    "charset": "utf8mb4",
    "connect_timeout": int(os.getenv("MYSQL_CONNECT_TIMEOUT", "30")),
    "read_timeout": int(os.getenv("MYSQL_READ_TIMEOUT", "60")),
    "write_timeout": int(os.getenv("MYSQL_WRITE_TIMEOUT", "60")),
}

# 诗歌表名（中央库中的诗歌表）
POEM_TABLE = os.getenv("POEM_TABLE", "quiz_poem_2")

# 地名提取参数（与 poem_test 一致）
DEFAULT_MODEL = os.getenv("PLACE_EXTRACT_MODEL", "deepseek-ai/DeepSeek-V3.2")
PROMPT_ID = int(os.getenv("PLACE_PROMPT_ID", "3"))  # 3=批量 JSON 模式
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "120"))
MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "1000"))
MAX_ITEMS_PER_BATCH = int(os.getenv("MAX_ITEMS_PER_BATCH", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# 无任务时轮询间隔（秒）
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
