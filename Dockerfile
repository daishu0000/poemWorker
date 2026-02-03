# 诗歌地名提取 Worker - 子系统任务处理
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# Worker 为长运行进程，无 HTTP 端口
CMD ["python", "worker.py"]
