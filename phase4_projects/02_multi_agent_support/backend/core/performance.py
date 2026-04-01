"""
性能优化模块

提供：
- 响应缓存（减少重复 LLM 调用）
- 会话持久化（SQLite）
- 连接池管理
- 请求限流
"""

import os
import sqlite3
import json
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional, List
import pickle

# ==================== 缓存管理 ====================

class ResponseCache:
    """
    响应缓存系统

    使用 TTL 缓存避免过期数据
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
        """
        self._cache: Dict[str, tuple] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _generate_key(self, prompt: str, agent_type: str = "default") -> str:
        """生成缓存键"""
        content = f"{agent_type}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, agent_type: str = "default") -> Optional[Dict[str, Any]]:
        """
        获取缓存的响应

        Args:
            prompt: 用户输入
            agent_type: 代理类型

        Returns:
            缓存的响应，如果不存在或已过期则返回 None
        """
        key = self._generate_key(prompt, agent_type)

        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                # 检查是否过期
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return value
                else:
                    # 过期，删除
                    del self._cache[key]

            self._misses += 1
            return None

    def set(self, prompt: str, response: Dict[str, Any], agent_type: str = "default"):
        """
        设置缓存

        Args:
            prompt: 用户输入
            response: 响应内容
            agent_type: 代理类型
        """
        key = self._generate_key(prompt, agent_type)

        with self._lock:
            # LRU 淘汰
            if len(self._cache) >= self._max_size:
                # 删除最旧的条目
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (response, time.time())

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "max_size": self._max_size
            }


# ==================== 会话持久化 ====================

class SessionManager:
    """
    会话管理器

    使用 SQLite 存储对话历史
    """

    def __init__(self, db_path: str = None):
        """
        初始化会话管理器

        Args:
            db_path: 数据库路径，默认为 backend/data/sessions.db
        """
        if db_path is None:
            # 默认路径
            backend_dir = Path(__file__).parent.parent
            data_dir = backend_dir / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "sessions.db"

        self._db_path = str(db_path)
        self._lock = Lock()
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    messages TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON sessions(created_at DESC)
            """)

            conn.commit()

    def create_session(self, session_id: str = None) -> str:
        """
        创建新会话

        Args:
            session_id: 会话ID，如果不提供则自动生成

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = hashlib.sha256(
                f"{time.time()}:{os.urandom(16)}".encode()
            ).hexdigest()[:32]

        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions (id, messages, metadata) VALUES (?, ?, ?)",
                    (session_id, "[]", "{}")
                )
                conn.commit()

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            会话数据，如果不存在则返回 None
        """
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT messages, metadata, created_at, updated_at FROM sessions WHERE id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return {
                    "id": session_id,
                    "messages": json.loads(row[0]),
                    "metadata": json.loads(row[1]),
                    "created_at": row[2],
                    "updated_at": row[3]
                }

    def update_session(self, session_id: str, messages: List[Dict], metadata: Dict = None):
        """
        更新会话

        Args:
            session_id: 会话ID
            messages: 消息列表
            metadata: 元数据（可选）
        """
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                messages_json = json.dumps(messages, ensure_ascii=False)
                metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

                conn.execute("""
                    UPDATE sessions
                    SET messages = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (messages_json, metadata_json, session_id))
                conn.commit()

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        列出最近的会话

        Args:
            limit: 返回的最大数量

        Returns:
            会话列表
        """
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, created_at, updated_at,
                           json_array_length(messages) as message_count
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))

                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "id": row[0],
                        "created_at": row[1],
                        "updated_at": row[2],
                        "message_count": row[3]
                    })

                return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否成功删除
        """
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,)
                )
                conn.commit()
                return cursor.rowcount > 0

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        清理旧会话

        Args:
            days: 保留天数

        Returns:
            删除的会话数量
        """
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM sessions
                    WHERE updated_at < datetime('now', '-' || ? || ' days')
                """, (days,))
                conn.commit()
                return cursor.rowcount


# ==================== 限流器 ====================

class RateLimiter:
    """
    简单的请求限流器

    使用滑动窗口算法
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        初始化限流器

        Args:
            max_requests: 时间窗口内最大请求数
            window_seconds: 时间窗口（秒）
        """
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: Dict[str, List[float]] = {}
        self._lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        """
        检查请求是否允许

        Args:
            client_id: 客户端标识（IP 地址或会话 ID）

        Returns:
            是否允许请求
        """
        now = time.time()

        with self._lock:
            # 初始化客户端记录
            if client_id not in self._requests:
                self._requests[client_id] = []

            # 清理过期记录
            self._requests[client_id] = [
                ts for ts in self._requests[client_id]
                if now - ts < self._window
            ]

            # 检查是否超过限制
            if len(self._requests[client_id]) >= self._max_requests:
                return False

            # 记录本次请求
            self._requests[client_id].append(now)
            return True

    def get_remaining(self, client_id: str) -> int:
        """
        获取剩余请求数

        Args:
            client_id: 客户端标识

        Returns:
            剩余请求数
        """
        with self._lock:
            if client_id not in self._requests:
                return self._max_requests

            now = time.time()
            # 清理过期记录
            self._requests[client_id] = [
                ts for ts in self._requests[client_id]
                if now - ts < self._window
            ]

            return max(0, self._max_requests - len(self._requests[client_id]))


# ==================== 装饰器 ====================

def cached_response(cache: ResponseCache, agent_type: str = "default"):
    """
    缓存响应装饰器

    Args:
        cache: 缓存实例
        agent_type: 代理类型

    Usage:
        @cached_response(cache, agent_type="tech_support")
        def handle_message(message: str) -> Dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(message: str, *args, **kwargs):
            # 尝试从缓存获取
            cached = cache.get(message, agent_type)
            if cached is not None:
                return cached

            # 调用原函数
            result = func(message, *args, **kwargs)

            # 存入缓存
            cache.set(message, result, agent_type)

            return result
        return wrapper
    return decorator


# ==================== 搜索缓存 ====================

class SearchCache:
    """
    搜索结果缓存

    缓存查询-结果对，避免重复检索和重排序。
    """

    def __init__(self, max_size: int = 200, ttl_seconds: int = 1800):
        """
        初始化搜索缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（默认 30 分钟）
        """
        self._cache: Dict[str, tuple] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, query: str, category: str = None, k: int = 3) -> str:
        return hashlib.sha256(f"{query}:{category}:{k}".encode()).hexdigest()

    def get(self, query: str, category: str = None, k: int = 3) -> Optional[List]:
        """获取缓存的搜索结果"""
        key = self._key(query, category, k)
        with self._lock:
            if key in self._cache:
                value, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._hits += 1
                    return value
                del self._cache[key]
            self._misses += 1
            return None

    def set(self, query: str, results: List, category: str = None, k: int = 3):
        """缓存搜索结果"""
        key = self._key(query, category, k)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest]
            self._cache[key] = (results, time.time())

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


# ==================== 全局实例 ====================

# 默认缓存实例（1000 条，1 小时过期）
default_cache = ResponseCache(max_size=1000, ttl_seconds=3600)

# 默认会话管理器
default_session_manager = SessionManager()

# 默认限流器（60 请求 / 分钟）
default_rate_limiter = RateLimiter(max_requests=60, window_seconds=60)

# 默认搜索缓存（200 条，30 分钟过期）
default_search_cache = SearchCache(max_size=200, ttl_seconds=1800)
