"""
模块 16：LangGraph 基础
学习如何使用 LangGraph 创建状态图工作流
"""

import os
import sys
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI）
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ============================================================
# 示例 1：简单顺序工作流
# ============================================================

def simple_workflow():
    """
    最简单的 LangGraph 示例：顺序执行的节点
    流程：START -> 预处理 -> LLM处理 -> 后处理 -> END
    """
    print("\n" + "=" * 60)
    print("示例 1：简单顺序工作流")
    print("=" * 60)

    # 定义状态
    class SimpleState(TypedDict):
        input_text: str
        processed_text: str
        llm_response: str
        final_output: str

    # 初始化模型
    
    # 定义节点函数
    def preprocess(state: SimpleState) -> dict:
        """预处理节点：清理和格式化输入"""
        text = state["input_text"].strip().lower()
        print(f"  [预处理] 输入: '{state['input_text']}' -> '{text}'")
        return {"processed_text": text}

    def call_llm(state: SimpleState) -> dict:
        """LLM 节点：调用语言模型"""
        messages = [
            SystemMessage(content="你是一个友好的助手，请简洁回答问题。"),
            HumanMessage(content=state["processed_text"])
        ]
        response = model.invoke(messages)
        print(f"  [LLM] 响应: {response.content[:50]}...")
        return {"llm_response": response.content}

    def postprocess(state: SimpleState) -> dict:
        """后处理节点：格式化输出"""
        final = f"✨ AI 回复：{state['llm_response']}"
        print("  [后处理] 完成格式化")
        return {"final_output": final}

    # 构建图
    graph = StateGraph(SimpleState)
    
    # 添加节点
    graph.add_node("preprocess", preprocess)
    graph.add_node("call_llm", call_llm)
    graph.add_node("postprocess", postprocess)
    
    # 添加边（定义执行顺序）
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "call_llm")
    graph.add_edge("call_llm", "postprocess")
    graph.add_edge("postprocess", END)
    
    # 编译图
    app = graph.compile()
    
    # 可视化图结构（打印）
    print("\n图结构：START -> preprocess -> call_llm -> postprocess -> END")
    
    # 运行
    result = app.invoke({"input_text": "  什么是人工智能？  "})
    
    print(f"\n最终输出：\n{result['final_output']}")
    
    return result

# ============================================================
# 示例 2：条件分支工作流
# ============================================================

def conditional_workflow():
    """
    带条件分支的工作流
    根据输入内容选择不同的处理路径
    """
    print("\n" + "=" * 60)
    print("示例 2：条件分支工作流")
    print("=" * 60)

    class ConditionalState(TypedDict):
        query: str
        query_type: str
        response: str

    
    def classify_query(state: ConditionalState) -> dict:
        """分类节点：判断查询类型"""
        query = state["query"].lower()
        
        if any(word in query for word in ["天气", "温度", "下雨"]):
            query_type = "weather"
        elif any(word in query for word in ["计算", "加", "减", "乘", "除", "等于"]):
            query_type = "math"
        else:
            query_type = "general"
        
        print(f"  [分类] 查询类型: {query_type}")
        return {"query_type": query_type}

    def handle_weather(state: ConditionalState) -> dict:
        """处理天气查询"""
        print("  [天气处理] 执行天气查询逻辑...")
        # 实际应用中这里会调用天气 API
        response = "🌤️ 今天天气晴朗，温度 25°C，适合外出！"
        return {"response": response}

    def handle_math(state: ConditionalState) -> dict:
        """处理数学计算"""
        print("  [数学处理] 执行计算逻辑...")
        messages = [
            SystemMessage(content="你是一个数学助手，请计算并给出结果。"),
            HumanMessage(content=state["query"])
        ]
        result = model.invoke(messages)
        return {"response": f"🔢 {result.content}"}

    def handle_general(state: ConditionalState) -> dict:
        """处理一般查询"""
        print("  [通用处理] 执行通用 LLM 调用...")
        messages = [
            SystemMessage(content="你是一个知识渊博的助手，请回答问题。"),
            HumanMessage(content=state["query"])
        ]
        result = model.invoke(messages)
        return {"response": f"💡 {result.content}"}

    def route_query(state: ConditionalState) -> Literal["weather", "math", "general"]:
        """路由函数：返回下一个节点名称"""
        return state["query_type"]

    # 构建图
    graph = StateGraph(ConditionalState)
    
    # 添加节点
    graph.add_node("classify", classify_query)
    graph.add_node("weather", handle_weather)
    graph.add_node("math", handle_math)
    graph.add_node("general", handle_general)
    
    # 添加边
    graph.add_edge(START, "classify")
    
    # 添加条件边
    graph.add_conditional_edges(
        "classify",  # 从哪个节点出发
        route_query,  # 路由函数
        {  # 路由映射
            "weather": "weather",
            "math": "math",
            "general": "general"
        }
    )
    
    # 所有处理节点都到 END
    graph.add_edge("weather", END)
    graph.add_edge("math", END)
    graph.add_edge("general", END)
    
    # 编译
    app = graph.compile()
    
    # 测试不同类型的查询
    test_queries = [
        "今天北京的天气怎么样？",
        "计算 123 加 456 等于多少？",
        "Python 是什么编程语言？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        result = app.invoke({"query": query})
        print(f"响应: {result['response'][:100]}...")
    
    return result

# ============================================================
# 示例 3：带内存的对话工作流
# ============================================================

def conversation_workflow():
    """
    带内存的多轮对话工作流
    使用 add_messages 注解自动管理消息历史
    """
    print("\n" + "=" * 60)
    print("示例 3：带内存的对话工作流")
    print("=" * 60)

    # 使用 add_messages 注解，消息会自动追加
    class ConversationState(TypedDict):
        messages: Annotated[list, add_messages]
        turn_count: int

    
    def chat_node(state: ConversationState) -> dict:
        """对话节点：处理用户消息并生成回复"""
        # 添加系统提示（如果是第一轮）
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(content="你是一个友好的中文助手。记住用户告诉你的信息。")
            ] + messages
        
        # 调用模型
        response = model.invoke(messages)
        
        # 更新轮数
        turn_count = state.get("turn_count", 0) + 1
        print(f"  [对话轮次 {turn_count}] AI: {response.content[:50]}...")
        
        # 返回新消息（会自动追加到 messages 列表）
        return {
            "messages": [response],  # add_messages 会自动追加
            "turn_count": turn_count
        }

    def should_continue(state: ConversationState) -> Literal["continue", "end"]:
        """决定是否继续对话"""
        # 这里简化为总是返回 end，实际应用中可以检查用户意图
        return "end"

    # 构建图
    graph = StateGraph(ConversationState)
    
    graph.add_node("chat", chat_node)
    
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)  # 简化：每次调用处理一轮
    
    # 使用内存保存器
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    # 模拟多轮对话（使用相同的 thread_id）
    config = {"configurable": {"thread_id": "user_001"}}
    
    conversations = [
        "你好！我叫小明。",
        "我最喜欢的编程语言是 Python。",
        "你还记得我的名字吗？",
        "我喜欢什么编程语言？"
    ]
    
    for user_input in conversations:
        print(f"\n用户: {user_input}")
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        print(f"AI: {result['messages'][-1].content}")
    
    # 查看完整对话历史
    print("\n" + "-" * 40)
    print("完整对话历史：")
    state = app.get_state(config)
    for msg in state.values["messages"]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content[:60]}...")
    
    return result

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("LangGraph 基础教程")
    print("=" * 60)
    
    # 运行示例
    simple_workflow()
    conditional_workflow()
    conversation_workflow()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
