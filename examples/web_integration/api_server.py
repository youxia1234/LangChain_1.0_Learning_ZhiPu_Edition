"""
LangChain Agent Web API 集成示例
演示如何将 LangChain Agent 部署为 REST API 服务

运行方式:
    uvicorn api_server:app --reload --port 8000

测试方式:
    # 单轮对话
    curl -X POST "http://localhost:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "北京天气如何？"}'

    # 多轮对话（带会话ID）
    curl -X POST "http://localhost:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "10 + 5", "session_id": "user123"}'

    # 流式输出
    curl -X POST "http://localhost:8000/chat/stream" \
        -H "Content-Type: application/json" \
        -d '{"message": "计算25乘以8"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
import json
from collections import defaultdict

# 加载环境变量
load_dotenv()

# =====================================
# 1. 定义 LangChain Agent
# =====================================

@tool
def search_database(query: str) -> str:
    """
    在内部数据库中搜索资料

    参数:
        query: 搜索关键词

    返回:
        搜索结果
    """
    # 模拟数据库搜索
    database = {
        "产品手册": "包含产品规格、使用说明、故障排除等内容",
        "技术文档": "API 文档、架构设计、部署指南",
        "培训材料": "新人培训、技能提升、最佳实践"
    }

    results = []
    for key, value in database.items():
        if query in key or query in value:
            results.append(f"{key}: {value}")

    if results:
        return "\n".join(results)
    else:
        return f"未找到关于'{query}'的资料"


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    执行数学计算

    参数:
        operation: 操作类型 (add, subtract, multiply, divide)
        a: 第一个数字
        b: 第二个数字

    返回:
        计算结果
    """
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "错误：除数不能为零"
            result = a / b
        else:
            return f"未知操作: {operation}"

        return f"{a} {operation} {b} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 创建 Agent
def create_ai_agent():
    """创建 LangChain Agent 实例"""
    model = ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.7
    )

    agent = create_agent(
        model=model,
        tools=[search_database, calculator],
        system_prompt="""你是一个智能资料检索助手。

你的主要功能：
1. 根据用户需求搜索相关资料
2. 执行必要的计算任务
3. 提供清晰准确的回答

工作流程：
1. 理解用户需求
2. 使用 search_database 工具搜索资料
3. 如需计算，使用 calculator 工具
4. 整合结果并给出友好回答

回答风格：
- 简洁明了
- 条理清晰
- 主动提供相关信息"""
    )

    return agent


# =====================================
# 2. FastAPI 应用
# =====================================

app = FastAPI(title="AI Assistant API", version="1.0.0")

# 全局 Agent 实例
agent = create_ai_agent()

# 会话存储（生产环境应使用 Redis 或数据库）
sessions: defaultdict = defaultdict(list)


# =====================================
# 3. 数据模型
# =====================================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str  # 用户消息
    session_id: Optional[str] = None  # 会话 ID（可选）


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str  # AI 回答
    session_id: str  # 会话 ID
    tools_used: List[str] = []  # 使用的工具


# =====================================
# 4. API 端点
# =====================================

@app.get("/")
async def root():
    """根路径 - API 信息"""
    return {
        "message": "AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - 单轮或多轮对话",
            "/chat/stream": "POST - 流式输出",
            "/sessions/{session_id}": "GET - 获取会话历史",
            "/sessions/{session_id}": "DELETE - 清除会话"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通聊天端点（非流式）

    返回完整回答，适合大多数应用场景
    """
    try:
        # 生成或获取会话 ID
        session_id = request.session_id or f"session_{hash(request.message)}"

        # 构建消息历史
        messages = sessions[session_id] + [
            {"role": "user", "content": request.message}
        ]

        # 调用 Agent
        response = agent.invoke({"messages": messages})

        # 更新会话历史
        sessions[session_id].extend(response["messages"])

        # 获取最终答案
        final_answer = response["messages"][-1].content

        # 提取使用的工具
        tools_used = []
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])

        return ChatResponse(
            answer=final_answer,
            session_id=session_id,
            tools_used=list(set(tools_used))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 错误: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天端点

    实时返回 Agent 的处理过程，适合需要显示进度的场景
    """
    session_id = request.session_id or f"session_{hash(request.message)}"

    messages = sessions[session_id] + [
        {"role": "user", "content": request.message}
    ]

    async def generate():
        """生成流式响应"""
        try:
            step_count = 0
            for chunk in agent.stream({"messages": messages}):
                step_count += 1

                if "messages" in chunk:
                    latest = chunk["messages"][-1]

                    # 发送状态更新
                    if hasattr(latest, "tool_calls") and latest.tool_calls:
                        for tc in latest.tool_calls:
                            data = {
                                "step": step_count,
                                "type": "tool_call",
                                "tool": tc["name"],
                                "args": tc["args"],
                                "content": None
                            }
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    elif hasattr(latest, "content") and latest.content:
                        data = {
                            "step": step_count,
                            "type": "answer",
                            "tool": None,
                            "args": None,
                            "content": latest.content
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # 更新会话（最后需要完整响应）
            # 注意：流式输出时，完整历史需要从 agent.invoke() 获取

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话历史"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 返回简化的历史（不包含系统消息）
    history = []
    for msg in sessions[session_id]:
        if hasattr(msg, "content"):
            history.append({
                "role": msg.__class__.__name__,
                "content": msg.content
            })

    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清除会话历史"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"会话 {session_id} 已清除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


# =====================================
# 5. 运行服务器
# =====================================

if __name__ == "__main__":
    import uvicorn

    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║       AI Assistant API Server is starting...          ║
    ╠═══════════════════════════════════════════════════════╣
    ║  API 文档: http://localhost:8000/docs                  ║
    ║  根路径: http://localhost:8000/                        ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
