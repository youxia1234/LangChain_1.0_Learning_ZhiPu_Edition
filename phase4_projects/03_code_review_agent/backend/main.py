"""FastAPI 后端应用

提供代码审查系统的 API 端点：
- POST /api/review — 提交代码审查请求
- GET /api/health — 健康检查
- POST /api/standards/upload — 上传自定义规范
- GET /api/standards/list — 列出规范库
- DELETE /api/standards/{source} — 删除规范文档
"""

import os
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from backend.models.schemas import (
    ReviewRequest,
    ReviewResponse,
    StandardDocument,
)
from backend.core.workflow import run_review
from backend.core.rag.standards_store import StandardsStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="AI Code Review Agent",
    description="基于 LangChain 1.0 + LangGraph 1.0 的智能代码审查系统",
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局规范知识库实例
_standards_store = None


def get_standards_store() -> StandardsStore:
    """获取规范知识库实例（单例）"""
    global _standards_store
    if _standards_store is None:
        _standards_store = StandardsStore()
        _standards_store.initialize()
    return _standards_store


# ==================== 审查 API ====================


@app.post("/api/review", response_model=ReviewResponse)
async def review_code(request: ReviewRequest):
    """提交代码审查请求

    分析指定 Git 仓库的 diff，返回结构化审查报告。

    Args:
        request: 审查请求参数

    Returns:
        ReviewResponse: 审查结果
    """
    logger.info(f"收到审查请求: {request.repo_path} (target: {request.target_branch})")

    # 验证仓库路径
    if not os.path.isdir(request.repo_path):
        raise HTTPException(status_code=400, detail=f"仓库路径不存在: {request.repo_path}")

    git_dir = os.path.join(request.repo_path, ".git")
    if not os.path.exists(git_dir):
        raise HTTPException(status_code=400, detail=f"不是有效的 Git 仓库: {request.repo_path}")

    try:
        review_config = {
            "enable_style": request.enable_style,
            "enable_security": request.enable_security,
            "enable_performance": request.enable_performance,
            "enable_architecture": request.enable_architecture,
        }

        report = await run_review(
            repo_path=request.repo_path,
            target_branch=request.target_branch,
            review_config=review_config,
        )

        if report.get("error"):
            return ReviewResponse(
                success=False,
                error=report["error"],
            )

        return ReviewResponse(
            success=True,
            report=_format_report(report),
        )

    except Exception as e:
        logger.error(f"审查执行失败: {e}", exc_info=True)
        return ReviewResponse(
            success=False,
            error=str(e),
        )


def _format_report(report: dict) -> dict:
    """格式化报告以符合 ReviewReport schema

    Args:
        report: 原始报告字典

    Returns:
        dict: 格式化后的报告
    """
    summary = report.get("summary", {})
    findings = report.get("findings", [])

    return {
        "repo_path": report.get("repo_path", ""),
        "target_branch": report.get("target_branch", "HEAD~1"),
        "summary": summary,
        "findings": findings,
    }


# ==================== 健康检查 ====================


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    store = get_standards_store()
    stats = store.get_stats()

    return {
        "status": "healthy",
        "version": "1.0.0",
        "standards_loaded": stats.get("total_chunks", 0),
        "categories": stats.get("categories", []),
    }


# ==================== 规范管理 API ====================


@app.post("/api/standards/upload")
async def upload_standard(
    file: UploadFile = File(...),
    category: str = Form(...),
):
    """上传自定义规范文档

    支持 .md, .txt, .pdf 格式。

    Args:
        file: 上传的文件
        category: 规范类别

    Returns:
        dict: 上传结果
    """
    # 验证文件类型
    allowed_extensions = {".md", ".txt", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，允许: {allowed_extensions}",
        )

    # 验证类别
    valid_categories = {"style", "security", "performance", "architecture", "custom"}
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"无效的类别: {category}，允许: {valid_categories}",
        )

    # 保存上传文件
    upload_dir = os.getenv("UPLOAD_DIR", "./data/uploads")
    if not os.path.isabs(upload_dir):
        base_dir = Path(__file__).parent.parent
        upload_dir = str(base_dir / upload_dir)

    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 添加到知识库
        store = get_standards_store()
        chunk_count = store.add_user_documents(file_path, category)

        return {
            "success": True,
            "filename": file.filename,
            "category": category,
            "chunk_count": chunk_count,
        }

    except Exception as e:
        logger.error(f"上传规范文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/standards/list")
async def list_standards():
    """列出知识库中的所有规范文档"""
    store = get_standards_store()
    docs = store.list_documents()
    stats = store.get_stats()

    return {
        "documents": docs,
        "total_chunks": stats.get("total_chunks", 0),
        "categories": stats.get("categories", []),
    }


@app.delete("/api/standards/{source}")
async def delete_standard(source: str):
    """删除指定的规范文档

    Args:
        source: 文档来源标识（文件名）
    """
    # 不允许删除内置规范
    builtin_files = {
        "python_style.md", "security_rules.md",
        "performance_patterns.md", "architecture_principles.md",
    }
    if source in builtin_files:
        raise HTTPException(status_code=403, detail="不能删除内置规范文档")

    store = get_standards_store()
    success = store.delete_document(source)

    if success:
        return {"success": True, "message": f"已删除: {source}"}
    else:
        raise HTTPException(status_code=404, detail=f"文档不存在: {source}")


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
