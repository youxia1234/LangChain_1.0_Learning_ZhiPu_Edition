"""
FastAPI 主程序

提供以下 API：
- POST /api/chat - 聊天接口
- POST /api/upload - 文档上传
- GET /api/knowledge - 知识库列表
- DELETE /api/knowledge/{doc_id} - 删除文档
- GET /api/stats - 统计信息
"""

import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import json

# 导入环境变量
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 禁用 LangSmith 监控（避免 API 错误）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# 导入核心模块
import sys
sys.path.append(str(Path(__file__).parent))

from core.agents import CustomerServiceSystem
from core.knowledge import get_kb_manager
from core.rag import create_rag_engine
from core.hybrid_rag import create_hybrid_rag_engine

# 导入模型
from models.chat import (
    ChatRequest,
    ChatResponse,
    SourceInfo,
    UploadResponse,
    DocumentInfo,
    KnowledgeListResponse,
    StatisticsResponse,
    DeleteResponse,
    ErrorResponse
)

# ==================== 创建 FastAPI 应用 ====================

app = FastAPI(
    title="多代理智能客服系统 API",
    description="基于 LangChain 1.0 + LangGraph 的多代理智能客服系统，支持 RAG 知识库检索",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 初始化系统 ====================

# 读取配置：是否启用混合检索
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))

# 初始化 RAG 引擎
rag_engine = None
try:
    if ENABLE_HYBRID_SEARCH:
        print(f"[RAG] Initializing HYBRID RAG engine (BM25 + Vector)...")
        print(f"[RAG] BM25 weight: {BM25_WEIGHT}, Vector weight: {VECTOR_WEIGHT}")
        rag_engine = create_hybrid_rag_engine(
            bm25_weight=BM25_WEIGHT,
            vector_weight=VECTOR_WEIGHT
        )
        print("[OK] HYBRID RAG engine initialized successfully")
        print("[OK] Supports BM25 keyword search + Vector semantic search")
    else:
        print("[RAG] Initializing standard RAG engine with Zhipu AI embeddings...")
        rag_engine = create_rag_engine()
        print("[OK] RAG engine initialized successfully")
except Exception as e:
    print(f"[WARN] RAG engine initialization failed: {e}")
    print("[WARN] System will run without RAG functionality")
    rag_engine = None

# 初始化多代理系统
print("[Agent] Initializing multi-agent system...")
cs_system = CustomerServiceSystem(rag_engine=rag_engine)

# 初始化知识库管理器（启用 RAG）
print("[KB] Initializing knowledge base manager...")
kb_manager = get_kb_manager(enable_rag=(rag_engine is not None))

print("[OK] System initialization complete")


# ==================== 健康检查 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "多代理智能客服系统 API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "upload": "/api/upload",
            "knowledge": "/api/knowledge",
            "stats": "/api/stats"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# ==================== 聊天接口 ====================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口（同步）

    处理用户消息，返回客服回复
    """
    try:
        result = cs_system.handle_message(request.message)

        # 转换来源信息
        sources = [
            SourceInfo(**source) for source in result.get("sources", [])
        ]

        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            confidence=result["confidence"],
            quality_score=result.get("quality_score", 0.0),
            escalated=result["escalated"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    聊天接口（流式输出）

    逐步返回响应内容，使用 Server-Sent Events (SSE) 格式
    """
    async def generate():
        try:
            # 获取流式生成器
            for chunk in cs_system.handle_message_stream(request.message):
                try:
                    # 将字典转换为 JSON 字符串
                    chunk_json = json.dumps(chunk, ensure_ascii=False)
                    # 使用 SSE 格式：data: <json>\n\n
                    yield f"data: {chunk_json}\n\n"
                except (TypeError, ValueError) as e:
                    # JSON 序列化失败，发送错误但继续处理
                    print(f"[ERROR] JSON 序列化失败: {e}, chunk: {chunk}")
                    error_chunk = {
                        "type": "error",
                        "error": f"数据序列化失败: {str(e)}",
                        "done": True
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                    break

        except Exception as e:
            # 发送错误信息
            print(f"[ERROR] 流式生成错误: {e}")
            import traceback
            traceback.print_exc()
            error_chunk = {
                "type": "error",
                "error": str(e),
                "done": True
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )


# ==================== 文档上传接口 ====================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="要上传的文档（PDF/TXT/MD）"),
    category: str = Form(..., description="知识库类别（products/technical/faq）")
):
    """
    上传文档到知识库

    支持的格式：PDF, TXT, Markdown
    """
    try:
        # 验证类别
        valid_categories = ["products", "technical", "faq"]
        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"无效的类别，必须是以下之一: {', '.join(valid_categories)}"
            )

        # 读取文件内容
        content = await file.read()

        # 保存文件
        filename = file.filename
        file_path = kb_manager.save_uploaded_file(content, filename, category)

        # 处理并索引到 Pinecone
        result = kb_manager.process_and_index_document(file_path, category)

        return UploadResponse(
            status="success",
            document_id=result["document_id"],
            chunks=result["chunks"],
            category=result["category"],
            source=result["source"],
            message=f"文档 '{filename}' 已成功添加到 {category} 知识库"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 知识库管理接口 ====================

@app.get("/api/knowledge", response_model=KnowledgeListResponse)
async def list_knowledge(category: str = None):
    """
    获取知识库文档列表

    Args:
        category: 可选，筛选特定类别
    """
    try:
        documents = kb_manager.list_documents(category)

        doc_list = [
            DocumentInfo(**doc) for doc in documents
        ]

        return KnowledgeListResponse(
            documents=doc_list,
            total=len(doc_list)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/knowledge/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """
    删除知识库中的文档

    Args:
        doc_id: 文档ID
    """
    try:
        success = kb_manager.delete_document(doc_id)

        if success:
            return DeleteResponse(
                status="success",
                message=f"文档 {doc_id} 已删除"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"文档 {doc_id} 不存在"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 统计接口 ====================

@app.get("/api/stats", response_model=StatisticsResponse)
async def get_statistics():
    """
    获取系统统计信息
    """
    try:
        stats = kb_manager.get_statistics()

        return StatisticsResponse(
            total_documents=stats["total_documents"],
            category_stats=stats["category_stats"],
            categories=stats["categories"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 主程序 ====================

def main():
    """运行 FastAPI 服务器"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print(f"\n{'='*60}")
    print(f"[START] Multi-Agent Customer Service System - FastAPI Backend")
    print(f"{'='*60}")
    print(f"\nServer starting...")
    print(f"URL: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"\nPress Ctrl+C to stop the server\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )


if __name__ == "__main__":
    main()
