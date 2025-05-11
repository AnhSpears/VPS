# backend/config.py
from pathlib import Path
import json

# Số lượt chat short-term
SHORT_TERM_LIMIT = 20

# Đường dẫn file logs (latency & errors)
LOG_FILE = Path(__file__).parent / "data" / "logs.json"
# Đường dẫn file lưu history dài hạn
CHAT_HISTORY_FILE = Path(__file__).parent / "data" / "chat_history.jsonl"

# Tạo thư mục data nếu chưa có
DATA_DIR = LOG_FILE.parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo file logs.json nếu chưa tồn tại
if not LOG_FILE.exists():
    LOG_FILE.write_text("[]", encoding="utf-8")
# Khởi tạo file chat_history.jsonl nếu chưa tồn tại
if not CHAT_HISTORY_FILE.exists():
    CHAT_HISTORY_FILE.write_text("", encoding="utf-8")
