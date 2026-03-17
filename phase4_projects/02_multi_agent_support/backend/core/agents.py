"""
多代理系统核心模块

本模块包含：
- IntentClassifier: 意图分类器
- TechSupportAgent: 技术支持代理（集成 RAG）
- OrderServiceAgent: 订单服务代理
- ProductConsultAgent: 产品咨询代理（集成 RAG）
- QualityChecker: 质量检查器
- CustomerServiceSystem: 系统编排器
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph
from langgraph.graph import StateGraph, START, END

# 加载环境变量
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# 禁用 LangSmith 监控（避免 API 错误）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取密钥"
    )

# 初始化模型（使用智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


# ==================== JSON 解析辅助函数 ====================

def safe_parse_json(text: str, default: dict = None) -> dict:
    """
    安全地解析JSON文本

    处理：
    - Markdown 代码块 (```json ... ```)
    - 前后的空白字符
    - 解析失败时返回默认值
    """
    if default is None:
        default = {}

    content = text.strip()

    # 移除 Markdown 代码块
    if "```json" in content:
        try:
            content = content.split("```json")[1].split("```")[0]
        except IndexError:
            pass
    elif "```" in content:
        try:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        except IndexError:
            pass

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return default


# ==================== 模拟数据库 ====================

MOCK_ORDERS = {
    "ORD001": {
        "status": "已发货",
        "product": "智能手表 Pro",
        "price": 1299,
        "shipping": "顺丰快递",
        "tracking": "SF1234567890",
        "estimated_delivery": "2024-12-20"
    },
    "ORD002": {
        "status": "处理中",
        "product": "无线耳机 Max",
        "price": 899,
        "shipping": "待发货",
        "tracking": None,
        "estimated_delivery": "2024-12-22"
    },
    "ORD003": {
        "status": "已完成",
        "product": "便携充电宝",
        "price": 199,
        "shipping": "已签收",
        "tracking": "YT9876543210",
        "estimated_delivery": "2024-12-15"
    }
}

MOCK_PRODUCTS = {
    "智能手表 Pro": {
        "price": 1299,
        "features": ["心率监测", "GPS定位", "防水50米", "7天续航"],
        "stock": 50,
        "rating": 4.8
    },
    "无线耳机 Max": {
        "price": 899,
        "features": ["主动降噪", "40小时续航", "蓝牙5.3", "通话降噪"],
        "stock": 120,
        "rating": 4.6
    },
    "便携充电宝": {
        "price": 199,
        "features": ["20000mAh", "快充支持", "双USB输出", "LED显示"],
        "stock": 200,
        "rating": 4.5
    },
    "智能音箱": {
        "price": 499,
        "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi音质"],
        "stock": 80,
        "rating": 4.7
    }
}


# ==================== 工具定义 ====================

@tool
def query_order(order_id: str) -> str:
    """查询订单信息

    Args:
        order_id: 订单号，格式如 ORD001

    Returns:
        订单详情的JSON字符串
    """
    order = MOCK_ORDERS.get(order_id.upper())
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)
    return f"未找到订单 {order_id}"

@tool
def track_shipping(tracking_number: str) -> str:
    """查询物流信息

    Args:
        tracking_number: 物流单号

    Returns:
        物流状态信息
    """
    # 模拟物流信息
    if tracking_number.startswith("SF"):
        return f"顺丰快递 {tracking_number}: 包裹已到达配送站，预计今日送达"
    elif tracking_number.startswith("YT"):
        return f"圆通快递 {tracking_number}: 已签收"
    return f"未找到物流信息 {tracking_number}"

@tool
def search_product(keyword: str) -> str:
    """搜索产品信息

    Args:
        keyword: 产品关键词

    Returns:
        匹配产品的信息
    """
    results = []
    for name, info in MOCK_PRODUCTS.items():
        if keyword.lower() in name.lower():
            results.append({
                "name": name,
                "price": f"¥{info['price']}",
                "features": info['features'],
                "rating": f"{info['rating']}分"
            })

    if results:
        return json.dumps(results, ensure_ascii=False, indent=2)
    return f"未找到包含 '{keyword}' 的产品"

@tool
def get_product_recommendations(budget: int, category: str = "全部") -> str:
    """根据预算推荐产品

    Args:
        budget: 预算金额
        category: 产品类别（可选）

    Returns:
        推荐产品列表
    """
    recommendations = []
    for name, info in MOCK_PRODUCTS.items():
        if info['price'] <= budget:
            recommendations.append({
                "name": name,
                "price": f"¥{info['price']}",
                "rating": info['rating']
            })

    # 按评分排序
    recommendations.sort(key=lambda x: float(x['rating']), reverse=True)

    if recommendations:
        return json.dumps(recommendations[:3], ensure_ascii=False, indent=2)
    return f"在预算 ¥{budget} 内暂无推荐产品"


# ==================== 状态定义 ====================

class CustomerServiceState(Dict):
    """客服系统状态"""
    user_message: str
    chat_history: List[Dict[str, str]]
    intent: str
    confidence: float
    agent_response: str
    needs_escalation: bool
    escalation_reason: str
    quality_score: float
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]  # RAG 检索来源


# ==================== 代理定义 ====================

class IntentClassifier:
    """意图分类器"""

    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图分类专家。分析用户消息并返回意图分类。

可选意图：
- tech_support: 技术问题、故障排除、使用帮助
- order_service: 订单查询、物流跟踪、退换货
- product_consult: 产品咨询、价格询问、功能介绍
- escalate: 投诉、无法理解、需要人工

返回格式（JSON）：
{{"intent": "意图类型", "confidence": 0.0-1.0, "reason": "分类原因"}}

只返回JSON，不要其他内容。"""),
            ("human", "{message}")
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        """分类用户意图"""
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({"message": message})

        # 使用安全的 JSON 解析
        default_result = {"intent": "escalate", "confidence": 0.5, "reason": "解析失败"}
        parsed = safe_parse_json(result, default_result)

        # 确保返回有效的意图
        if "intent" not in parsed:
            return default_result
        return parsed


class TechSupportAgent:
    """技术支持代理（集成 RAG）"""

    def __init__(self, rag_engine=None):
        self.llm = model
        self.rag_engine = rag_engine

        # 基础工具
        tools = []

        # 如果有 RAG 引擎，添加检索工具
        if self.rag_engine:
            @tool
            def search_technical_knowledge(query: str) -> str:
                """从技术文档知识库中搜索解决方案"""
                try:
                    results = self.rag_engine.search(query, category="technical", k=3)
                    return "\n\n".join([doc.page_content for doc in results])
                except Exception as e:
                    return f"知识库检索失败: {str(e)}"

            tools.append(search_technical_knowledge)

        self.system_prompt = """你是一个专业的技术支持工程师。你的职责是：
1. 分析用户遇到的技术问题
2. 使用 search_technical_knowledge 工具从技术文档中查找解决方案
3. 提供清晰的故障排除步骤
4. 如果问题超出能力范围，建议升级到人工支持

回复要求：
- 语气友好专业
- 步骤清晰有序
- 提供多个可能的解决方案"""

        # 创建 agent
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        """处理技术支持请求（同步）"""
        messages = [{"role": "user", "content": message}]

        result = self.agent.invoke({"messages": messages})

        # 提取最终回复
        if result["messages"]:
            response = result["messages"][-1].content

            # 检查是否有工具调用（RAG 检索）
            sources = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        if "name" in call and call["name"] == "search_technical_knowledge":
                            sources.append({
                                "type": "knowledge_base",
                                "tool": "search_technical_knowledge",
                                "query": call["args"].get("query", "")
                            })

            return {
                "response": response,
                "sources": sources
            }

        return {
            "response": "抱歉，我暂时无法处理您的问题。建议联系人工客服。",
            "sources": []
        }

    def handle_stream(self, message: str, chat_history: List = None):
        """
        处理技术支持请求（流式输出）

        使用 LLM 的 .stream() 方法逐 token 生成响应
        """
        try:
            # 构建提示词（包含 RAG 上下文）
            prompt = self.system_prompt + "\n\n"

            # 如果有 RAG 引擎，检索相关文档
            if self.rag_engine:
                search_results = self.rag_engine.search(message, k=3)
                if search_results:
                    prompt += "参考信息：\n"
                    for result in search_results:
                        prompt += f"- {result.page_content}\n"
                    prompt += "\n"

            prompt += f"用户问题：{message}"

            # 使用 LLM 的 stream 方法获取逐 token 输出
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，出现了一些问题。请稍后再试。"


class OrderServiceAgent:
    """订单服务代理"""

    def __init__(self, rag_engine=None):
        self.llm = model
        self.tools = [query_order, track_shipping]

        self.system_prompt = """你是一个专业的订单服务专员。你的职责是：
1. 帮助用户查询订单状态
2. 提供物流跟踪信息
3. 解答退换货相关问题
4. 使用工具获取准确信息

回复要求：
- 信息准确完整
- 主动提供相关信息
- 如果需要订单号，礼貌询问"""

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        """处理订单服务请求（同步）"""
        messages = [{"role": "user", "content": message}]

        result = self.agent.invoke({"messages": messages})

        if result["messages"]:
            return {
                "response": result["messages"][-1].content,
                "sources": []
            }

        return {
            "response": "抱歉，订单查询服务暂时不可用。请稍后再试。",
            "sources": []
        }

    def handle_stream(self, message: str, chat_history: List = None):
        """
        处理订单服务请求（流式输出）

        使用 LLM 的 .stream() 方法逐 token 生成响应
        """
        try:
            # 构建提示词
            prompt = self.system_prompt + "\n\n"
            prompt += f"用户问题：{message}"

            # 使用 LLM 的 stream 方法获取逐 token 输出
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，出现了一些问题。请稍后再试。"


class ProductConsultAgent:
    """产品咨询代理（集成 RAG）"""

    def __init__(self, rag_engine=None):
        self.llm = model
        self.rag_engine = rag_engine

        # 基础工具
        tools = [search_product, get_product_recommendations]

        # 如果有 RAG 引擎，添加产品文档检索
        if self.rag_engine:
            @tool
            def search_product_docs(query: str) -> str:
                """从产品文档知识库中搜索信息"""
                try:
                    results = self.rag_engine.search(query, category="products", k=3)
                    return "\n\n".join([doc.page_content for doc in results])
                except Exception as e:
                    return f"产品文档检索失败: {str(e)}"

            tools.append(search_product_docs)

        self.system_prompt = """你是一个热情的产品顾问。你的职责是：
1. 介绍产品功能和特点
2. 使用 search_product_docs 工具从产品文档中查找详细信息
3. 根据用户需求推荐合适的产品
4. 使用工具获取最新产品信息

回复要求：
- 热情有亲和力
- 突出产品优势
- 根据用户需求推荐
- 不要过度推销"""

        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        """处理产品咨询请求（同步）"""
        messages = [{"role": "user", "content": message}]

        result = self.agent.invoke({"messages": messages})

        if result["messages"]:
            response = result["messages"][-1].content

            # 检查是否有工具调用（RAG 检索）
            sources = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        if "name" in call and call["name"] in ["search_product_docs", "search_product", "get_product_recommendations"]:
                            sources.append({
                                "type": "knowledge_base" if call["name"] == "search_product_docs" else "tool",
                                "tool": call["name"],
                                "query": call["args"].get("query", call["args"].get("keyword", ""))
                            })

            return {
                "response": response,
                "sources": sources
            }

        return {
            "response": "抱歉，产品信息查询暂时不可用。请稍后再试。",
            "sources": []
        }

    def handle_stream(self, message: str, chat_history: List = None):
        """
        处理产品咨询请求（流式输出）

        使用 LLM 的 .stream() 方法逐 token 生成响应
        """
        try:
            # 构建提示词（包含 RAG 上下文）
            prompt = self.system_prompt + "\n\n"

            # 如果有 RAG 引擎，检索相关文档
            if self.rag_engine:
                search_results = self.rag_engine.search(message, category="products", k=3)
                if search_results:
                    prompt += "参考信息：\n"
                    for result in search_results:
                        prompt += f"- {result.page_content}\n"
                    prompt += "\n"

            prompt += f"用户问题：{message}"

            # 使用 LLM 的 stream 方法获取逐 token 输出
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，出现了一些问题。请稍后再试。"


class QualityChecker:
    """质量检查器"""

    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是客服质量检查专家。评估客服回复的质量。

评估维度：
1. 相关性（0-25分）：回复是否针对用户问题
2. 完整性（0-25分）：是否提供了足够的信息
3. 专业性（0-25分）：语言是否专业得体
4. 有用性（0-25分）：是否真正帮助到用户

返回格式（JSON）：
{{"total_score": 0-100, "needs_escalation": true/false, "reason": "评估说明"}}

只返回JSON。"""),
            ("human", """用户问题：{user_message}
客服回复：{agent_response}

请评估：""")
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        """检查回复质量"""
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "user_message": user_message,
            "agent_response": agent_response
        })

        # 使用安全的 JSON 解析
        default_result = {"total_score": 60, "needs_escalation": False, "reason": "评估完成"}
        return safe_parse_json(result, default_result)


# ==================== 客服系统主类 ====================

class CustomerServiceSystem:
    """多代理客服系统"""

    def __init__(self, rag_engine=None):
        # 初始化组件
        self.classifier = IntentClassifier()
        self.tech_agent = TechSupportAgent(rag_engine)
        self.order_agent = OrderServiceAgent(rag_engine)
        self.product_agent = ProductConsultAgent(rag_engine)
        self.quality_checker = QualityChecker()

        # 构建工作流图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""

        def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
            """分类用户意图"""
            print("[SEARCH] 分析用户意图...")
            result = self.classifier.classify(state["user_message"])

            state["intent"] = result.get("intent", "escalate")
            state["confidence"] = result.get("confidence", 0.5)

            print(f"   意图: {state['intent']} (置信度: {state['confidence']:.2f})")
            return state

        def route_to_agent(state: CustomerServiceState) -> Literal["tech_support", "order_service", "product_consult", "escalate"]:
            """路由到对应代理"""
            intent = state["intent"]
            confidence = state["confidence"]

            # 低置信度直接升级
            if confidence < 0.6:
                return "escalate"

            if intent == "tech_support":
                return "tech_support"
            elif intent == "order_service":
                return "order_service"
            elif intent == "product_consult":
                return "product_consult"
            else:
                return "escalate"

        def tech_support_handler(state: CustomerServiceState) -> CustomerServiceState:
            """技术支持处理"""
            print("[TOOL] 技术支持代理处理中...")
            result = self.tech_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def order_service_handler(state: CustomerServiceState) -> CustomerServiceState:
            """订单服务处理"""
            print("[PACKAGE] 订单服务代理处理中...")
            result = self.order_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def product_consult_handler(state: CustomerServiceState) -> CustomerServiceState:
            """产品咨询处理"""
            print("[SHOP] 产品咨询代理处理中...")
            result = self.product_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def escalate_handler(state: CustomerServiceState) -> CustomerServiceState:
            """升级处理"""
            print("[PERSON] 升级到人工客服...")
            state["needs_escalation"] = True
            state["escalation_reason"] = "意图识别置信度低或用户要求人工服务"
            state["agent_response"] = """非常抱歉，您的问题需要人工客服来处理。

我已经为您转接人工客服，请稍候...

在等待期间，您也可以：
1. 拨打客服热线：400-xxx-xxxx
2. 发送邮件至：support@example.com
3. 工作日 9:00-18:00 在线客服响应更快

感谢您的耐心等待！"""
            state["sources"] = []
            return state

        def quality_check(state: CustomerServiceState) -> CustomerServiceState:
            """质量检查"""
            print("[OK] 执行质量检查...")
            result = self.quality_checker.check(
                state["user_message"],
                state["agent_response"]
            )

            state["quality_score"] = result.get("total_score", 0) / 100

            # 质量太低需要升级
            if result.get("needs_escalation", False) or state["quality_score"] < 0.6:
                state["needs_escalation"] = True
                state["escalation_reason"] = result.get("reason", "质量检查未通过")

            print(f"   质量评分: {state['quality_score']:.2f}")
            return state

        def should_escalate(state: CustomerServiceState) -> Literal["escalate_final", "respond"]:
            """判断是否需要升级"""
            if state.get("needs_escalation", False):
                return "escalate_final"
            return "respond"

        def final_escalate(state: CustomerServiceState) -> CustomerServiceState:
            """最终升级处理"""
            # 保留原始回复但添加升级提示
            original_response = state["agent_response"]
            state["agent_response"] = f"""{original_response}

---
[WARN] 系统提示：由于此问题可能需要更专业的处理，我们建议您联系人工客服以获得更好的服务。"""
            return state

        def respond(state: CustomerServiceState) -> CustomerServiceState:
            """最终响应"""
            return state

        # 构建图
        graph = StateGraph(CustomerServiceState)

        # 添加节点
        graph.add_node("classify", classify_intent)
        graph.add_node("tech_support", tech_support_handler)
        graph.add_node("order_service", order_service_handler)
        graph.add_node("product_consult", product_consult_handler)
        graph.add_node("escalate", escalate_handler)
        graph.add_node("quality_check", quality_check)
        graph.add_node("escalate_final", final_escalate)
        graph.add_node("respond", respond)

        # 添加边
        graph.add_edge(START, "classify")

        # 条件路由
        graph.add_conditional_edges(
            "classify",
            route_to_agent,
            {
                "tech_support": "tech_support",
                "order_service": "order_service",
                "product_consult": "product_consult",
                "escalate": "escalate"
            }
        )

        # 代理处理后进行质量检查
        graph.add_edge("tech_support", "quality_check")
        graph.add_edge("order_service", "quality_check")
        graph.add_edge("product_consult", "quality_check")
        graph.add_edge("escalate", END)

        # 质量检查后的条件路由
        graph.add_conditional_edges(
            "quality_check",
            should_escalate,
            {
                "escalate_final": "escalate_final",
                "respond": "respond"
            }
        )

        graph.add_edge("escalate_final", END)
        graph.add_edge("respond", END)

        return graph.compile()

    def handle_message(self, message: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """处理用户消息（同步）"""
        print(f"\n{'='*60}")
        print(f"[CHAT] 用户: {message}")
        print('='*60)

        initial_state = CustomerServiceState(
            user_message=message,
            chat_history=chat_history or [],
            intent="",
            confidence=0.0,
            agent_response="",
            needs_escalation=False,
            escalation_reason="",
            quality_score=0.0,
            metadata={"timestamp": datetime.now().isoformat()},
            sources=[]
        )

        result = self.graph.invoke(initial_state)

        return {
            "response": result["agent_response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "quality_score": result["quality_score"],
            "escalated": result["needs_escalation"],
            "sources": result.get("sources", [])
        }

    def handle_message_stream(self, message: str, chat_history: List[Dict] = None):
        """
        处理用户消息（真正的流式输出）

        逐 token 生成响应，实现打字机效果
        """
        print(f"\n{'='*60}")
        print(f"[CHAT STREAM] 用户: {message}")
        print('='*60)

        # 1. 首先发送意图识别结果
        result = self.classifier.classify(message)
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.0)

        yield {
            "type": "intent",
            "intent": intent,
            "confidence": confidence,
            "done": False
        }

        # 2. 根据意图选择对应的代理进行流式输出
        try:
            if intent == "tech_support":
                # 使用技术支持代理的流式方法
                for chunk in self.tech_agent.handle_stream(message):
                    yield {
                        "type": "content",
                        "content": chunk,
                        "done": False
                    }

            elif intent == "order_service":
                # 使用订单服务代理的流式方法
                for chunk in self.order_agent.handle_stream(message):
                    yield {
                        "type": "content",
                        "content": chunk,
                        "done": False
                    }

            elif intent == "product_consult":
                # 使用产品咨询代理的流式方法
                for chunk in self.product_agent.handle_stream(message):
                    yield {
                        "type": "content",
                        "content": chunk,
                        "done": False
                    }

            else:
                # 升级到人工客服
                escalate_message = """非常抱歉，您的问题需要人工客服来处理。

我已经为您转接人工客服，请稍候...

在等待期间，您也可以：
1. 拨打客服热线：400-xxx-xxxx
2. 发送邮件至：support@example.com
3. 工作日 9:00-18:00 在线客服响应更快

感谢您的耐心等待！"""

                # 分段发送升级消息
                for i in range(0, len(escalate_message), 15):
                    chunk = escalate_message[i:i + 15]
                    yield {
                        "type": "content",
                        "content": chunk,
                        "done": False
                    }

            # 3. 获取完整结果（用于元数据）
            final_result = self.graph.invoke(
                CustomerServiceState(
                    user_message=message,
                    chat_history=chat_history or [],
                    intent=intent,
                    confidence=confidence,
                    agent_response="",
                    needs_escalation=False,
                    escalation_reason="",
                    quality_score=0.0,
                    metadata={"timestamp": datetime.now().isoformat()},
                    sources=[]
                )
            )

            # 4. 发送最终结果
            # 确保 sources 可以被 JSON 序列化
            sources = final_result.get("sources", [])
            # 将 sources 转换为可序列化的格式
            serializable_sources = []
            for source in sources:
                if isinstance(source, dict):
                    serializable_sources.append(source)
                else:
                    # 如果是 Document 对象或其他对象，转换为字典
                    serializable_sources.append({
                        "type": str(type(source).__name__),
                        "content": str(source) if hasattr(source, 'page_content') else str(source)
                    })

            yield {
                "type": "final",
                "response": final_result["agent_response"],
                "intent": final_result["intent"],
                "confidence": final_result["confidence"],
                "quality_score": final_result["quality_score"],
                "escalated": final_result["needs_escalation"],
                "sources": serializable_sources,
                "done": True
            }

        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            # 发生错误时返回错误消息
            yield {
                "type": "content",
                "content": "抱歉，处理您的请求时出现了错误。请稍后再试。",
                "done": False
            }

            # 发送完成标记
            yield {
                "type": "error",
                "error": str(e),
                "done": True
            }
