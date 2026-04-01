"""
企业智能问答系统 - 多代理核心模块

面向制造企业官网场景，包含：
- IntentClassifier: 意图分类器
- ProductInfoAgent: 产品咨询代理（集成 RAG + 查询重写 + 重排序）
- TechCapabilityAgent: 技术实力代理（集成 RAG）
- CompanyOverviewAgent: 公司概况代理
- PartnershipAgent: 合作咨询代理
- QualityChecker: 质量检查器
- EnterpriseQASystem: 系统编排器（LangGraph 工作流）
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

from .performance import default_cache

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取密钥"
    )

# 初始化模型
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


# ==================== JSON 解析辅助 ====================

def safe_parse_json(text: str, default: dict = None) -> dict:
    """安全解析 JSON 文本"""
    if default is None:
        default = {}

    content = text.strip()

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


# ==================== 制造企业模拟数据 ====================

MOCK_COMPANY = {
    "name": "华智精密制造有限公司",
    "name_en": "Huazhi Precision Manufacturing Co., Ltd.",
    "founded": 2008,
    "employees": 1200,
    "factory_area": "35000平方米",
    "annual_revenue": "8.5亿元",
    "location": "广东省深圳市光明新区科技产业园",
    "description": "华智精密制造是一家专注于智能硬件和精密电子元器件研发与生产的高新技术企业，拥有16年行业经验。",
    "core_advantages": [
        "16年精密制造经验",
        "1200+专业团队",
        "全自动化生产线",
        "ISO9001/IATF16949/CE/RoHS认证",
        "年产能5000万件",
        "7天快速打样"
    ],
    "milestones": [
        {"year": 2008, "event": "公司成立，专注精密模具设计"},
        {"year": 2012, "event": "通过 ISO9001 认证，进入消费电子领域"},
        {"year": 2015, "event": "建成全自动化产线，产能提升5倍"},
        {"year": 2018, "event": "通过 IATF16949 认证，进入汽车电子供应链"},
        {"year": 2020, "event": "获得国家高新技术企业认定"},
        {"year": 2023, "event": "年产值突破8亿元，员工超1200人"},
    ]
}

MOCK_PRODUCTS = {
    "HZ-WB200": {
        "name": "华智智能手表 WB200",
        "category": "智能穿戴",
        "price_range": "¥180-320",
        "moq": 1000,
        "description": "高精度健康监测智能手表，支持心率/血氧/GPS，IP68防水",
        "specs": {
            "屏幕": "1.85英寸 AMOLED",
            "防水": "IP68 (50米)",
            "续航": "14天",
            "传感器": "心率+血氧+加速度+陀螺仪",
            "连接": "蓝牙5.3 / NFC",
            "定位": "GPS+北斗+GLONASS"
        },
        "certifications": ["CE", "FCC", "RoHS", "REACH"],
        "applications": ["品牌代工", "企业定制", "健康养老", "运动健身"],
    },
    "HZ-EP500": {
        "name": "华智无线耳机 EP500",
        "category": "智能音频",
        "price_range": "¥120-250",
        "moq": 2000,
        "description": "主动降噪 TWS 耳机，40小时续航，Hi-Res 认证音质",
        "specs": {
            "降噪": "ANC 主动降噪 -42dB",
            "续航": "单次8h / 总40h",
            "蓝牙": "5.3 LE Audio",
            "音频": "Hi-Res / LDAC",
            "防水": "IPX5",
            "充电": "无线充电 / USB-C"
        },
        "certifications": ["CE", "FCC", "RoHS", "BQB"],
        "applications": ["品牌代工", "促销礼品", "电商专供"],
    },
    "HZ-PC100": {
        "name": "华智快充充电宝 PC100",
        "category": "智能充电",
        "price_range": "¥55-95",
        "moq": 3000,
        "description": "20000mAh 大容量，支持 65W PD 快充，三口输出",
        "specs": {
            "容量": "20000mAh (74Wh)",
            "输入": "USB-C 65W PD",
            "输出": "USB-C 65W + USB-A 22.5W ×2",
            "电池": "21700 锂电池",
            "重量": "380g",
            "安全": "过充/过放/短路/温控保护"
        },
        "certifications": ["CE", "FCC", "RoHS", "PSE", "CQC"],
        "applications": ["品牌代工", "企业礼品", "跨境电商"],
    },
    "HZ-TH800": {
        "name": "华智工业温湿度传感器 TH800",
        "category": "工业传感器",
        "price_range": "¥35-68",
        "moq": 5000,
        "description": "高精度工业级温湿度传感器，RS485/Modbus 通信，IP65 防护",
        "specs": {
            "温度范围": "-40°C ~ +125°C",
            "精度": "±0.3°C / ±2%RH",
            "通信": "RS485 / Modbus RTU",
            "供电": "DC 12-24V",
            "防护": "IP65",
            "接口": "M12 航空插头"
        },
        "certifications": ["CE", "RoHS", "FCC"],
        "applications": ["工业自动化", "冷链监控", "智慧农业", "环境监测"],
    },
    "HZ-AC300": {
        "name": "华智智能网关 AC300",
        "category": "物联网设备",
        "price_range": "¥150-280",
        "moq": 500,
        "description": "工业物联网边缘网关，支持 WiFi/4G/LoRa/以太网，边缘计算",
        "specs": {
            "处理器": "ARM Cortex-A7 1.2GHz",
            "内存": "512MB DDR3",
            "连接": "WiFi/4G/LoRa/以太网/RS485",
            "协议": "MQTT/Modbus/OPC UA",
            "供电": "DC 9-36V",
            "工作温度": "-40°C ~ +75°C"
        },
        "certifications": ["CE", "FCC", "RoHS"],
        "applications": ["智慧工厂", "智慧农业", "智慧城市", "能源管理"],
    },
}

MOCK_CERTIFICATIONS = {
    "ISO9001": {
        "name": "ISO 9001:2015 质量管理体系",
        "issued_by": "SGS",
        "valid_until": "2026-12",
        "scope": "精密电子元器件的设计、生产和销售"
    },
    "IATF16949": {
        "name": "IATF 16949:2016 汽车质量管理体系",
        "issued_by": "TÜV Rheinland",
        "valid_until": "2027-03",
        "scope": "汽车电子零部件的设计与制造"
    },
    "CE": {
        "name": "CE 欧盟符合性认证",
        "scope": "电磁兼容性 (EMC) + 低电压指令 (LVD)"
    },
    "RoHS": {
        "name": "RoHS 2.0 有害物质限制",
        "scope": "电子电气设备有害物质控制"
    },
    "HighTech": {
        "name": "国家高新技术企业",
        "issued_by": "广东省科技厅",
        "valid_until": "2025-09"
    },
}

MOCK_PARTNERSHIP = {
    "oem": {
        "name": "OEM 代工服务",
        "description": "客户提供设计图纸/方案，我们负责生产制造、品质管控和物流配送",
        "moq": "按产品类型不同，最低 500 件起",
        "lead_time": "打样7天，量产15-30天",
        "advantages": ["灵活起订量", "快速打样", "严格品控", "一站式物流"]
    },
    "odm": {
        "name": "ODM 设计制造",
        "description": "根据客户需求，提供从产品定义、ID设计、结构设计到生产制造的全流程服务",
        "moq": "按项目复杂度协商",
        "lead_time": "设计评估7天，首样20天，量产30天",
        "advantages": ["自有设计团队", "专利共享", "定制化功能", "品牌授权"]
    },
    "custom": {
        "name": "深度定制合作",
        "description": "针对企业级客户，提供软硬件一体化定制方案，含APP开发、云平台对接",
        "process": ["需求分析", "方案设计", "原型确认", "试产验证", "批量交付", "售后服务"],
        "advantages": ["软硬件一体化", "专属项目经理", "售后技术支持", "数据安全合规"]
    }
}


# ==================== 工具定义 ====================

@tool
def search_product_catalog(keyword: str) -> str:
    """搜索产品目录，按关键词查找匹配的产品信息

    Args:
        keyword: 产品关键词（型号、类别或功能）

    Returns:
        匹配产品的详细信息
    """
    results = []
    for model_id, info in MOCK_PRODUCTS.items():
        search_text = f"{model_id} {info['name']} {info['category']} {info['description']} {' '.join(info['applications'])}"
        if keyword.lower() in search_text.lower():
            results.append({
                "model": model_id,
                "name": info["name"],
                "category": info["category"],
                "price_range": info["price_range"],
                "moq": info["moq"],
                "description": info["description"],
                "certifications": info["certifications"],
            })

    if results:
        return json.dumps(results, ensure_ascii=False, indent=2)
    return f"未找到包含 '{keyword}' 的产品，您可以浏览我们的智能穿戴、智能音频、智能充电、工业传感器、物联网设备等品类。"


@tool
def get_certification_info(cert_name: str = "") -> str:
    """查询公司资质认证信息

    Args:
        cert_name: 认证名称（如 ISO9001、CE、RoHS），留空返回所有认证

    Returns:
        认证详细信息
    """
    if cert_name:
        for key, info in MOCK_CERTIFICATIONS.items():
            if cert_name.upper() in key.upper() or cert_name.upper() in info["name"].upper():
                return json.dumps(info, ensure_ascii=False, indent=2)
        return f"未找到 '{cert_name}' 相关的认证信息"

    return json.dumps(MOCK_CERTIFICATIONS, ensure_ascii=False, indent=2)


@tool
def compare_products(model_a: str, model_b: str) -> str:
    """对比两个产品的规格参数

    Args:
        model_a: 产品A型号（如 HZ-WB200）
        model_b: 产品B型号（如 HZ-EP500）

    Returns:
        产品对比信息
    """
    product_a = MOCK_PRODUCTS.get(model_a.upper())
    product_b = MOCK_PRODUCTS.get(model_b.upper())

    if not product_a and not product_b:
        return f"未找到型号 {model_a} 和 {model_b}"
    if not product_a:
        return f"未找到型号 {model_a}，可选型号: {', '.join(MOCK_PRODUCTS.keys())}"
    if not product_b:
        return f"未找到型号 {model_b}，可选型号: {', '.join(MOCK_PRODUCTS.keys())}"

    comparison = {
        model_a: {"name": product_a["name"], "specs": product_a["specs"], "price_range": product_a["price_range"]},
        model_b: {"name": product_b["name"], "specs": product_b["specs"], "price_range": product_b["price_range"]},
    }
    return json.dumps(comparison, ensure_ascii=False, indent=2)


@tool
def get_solution_recommendation(application: str) -> str:
    """根据应用场景推荐解决方案和产品

    Args:
        application: 应用场景（如 智慧工厂、品牌代工、健康养老、企业礼品）

    Returns:
        推荐的解决方案和产品列表
    """
    recommendations = []
    for model_id, info in MOCK_PRODUCTS.items():
        if application.lower() in " ".join(info["applications"]).lower():
            recommendations.append({
                "model": model_id,
                "name": info["name"],
                "category": info["category"],
                "price_range": info["price_range"],
                "moq": info["moq"],
                "why": f"适用于{application}场景，{info['description']}"
            })

    if recommendations:
        return json.dumps(recommendations, ensure_ascii=False, indent=2)
    return f"暂无针对 '{application}' 场景的现成方案，请联系我们的业务团队进行定制化方案设计。"


# ==================== 状态定义 ====================

class EnterpriseQAState(Dict):
    """企业问答系统状态"""
    user_message: str
    chat_history: List[Dict[str, str]]
    intent: str
    confidence: float
    agent_response: str
    needs_escalation: bool
    escalation_reason: str
    quality_score: float
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]


# ==================== RAG 辅助函数 ====================

def _build_rag_context(rag_engine, query: str, category: str = None, k: int = 3) -> tuple:
    """
    构建 RAG 上下文（含引用溯源信息）

    Returns:
        (context_text, sources) 元组
    """
    if not rag_engine:
        return "", []

    try:
        # 尝试混合检索
        if hasattr(rag_engine, 'search_hybrid'):
            results = rag_engine.search_hybrid(query, category=category, k=k)
        else:
            results = rag_engine.search(query, category=category, k=k)

        if not results:
            return "", []

        context_parts = []
        sources = []
        for i, doc in enumerate(results):
            source_name = doc.metadata.get("source", "知识库")
            chunk_idx = doc.metadata.get("chunk_index", "")
            context_parts.append(f"[来源{i+1}: {source_name}]{doc.page_content}")
            sources.append({
                "type": "knowledge_base",
                "source": source_name,
                "chunk_index": chunk_idx,
                "preview": doc.page_content[:100] + "..."
            })

        return "\n\n".join(context_parts), sources

    except Exception as e:
        print(f"[RAG] 检索失败: {e}")
        return "", []


# ==================== 代理定义 ====================

class IntentClassifier:
    """意图分类器 — 企业问答场景"""

    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图分类专家。分析访客对制造企业的提问，返回意图分类。

可选意图：
- product_info: 产品咨询、型号查询、参数对比、价格、选型
- tech_capability: 技术实力、生产工艺、质量管控、认证资质、研发能力
- company_overview: 公司介绍、发展历程、规模、企业文化、荣誉
- partnership: 合作咨询、代理政策、OEM/ODM、定制需求、商务对接
- escalate: 超出范围、投诉、需要人工接待

返回格式（JSON）：
{{"intent": "意图类型", "confidence": 0.0-1.0, "reason": "分类原因"}}

只返回JSON。"""),
            ("human", "{message}")
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({"message": message})
        default = {"intent": "company_overview", "confidence": 0.5, "reason": "解析失败"}
        parsed = safe_parse_json(result, default)
        if "intent" not in parsed:
            return default
        return parsed


class ProductInfoAgent:
    """产品咨询代理（集成 RAG + 查询重写）"""

    def __init__(self, rag_engine=None, query_processor=None):
        self.llm = model
        self.rag_engine = rag_engine
        self.query_processor = query_processor

        tools = [search_product_catalog, compare_products, get_solution_recommendation]

        if self.rag_engine:
            agent_self = self

            @tool
            def search_product_docs(query: str) -> str:
                """从产品文档知识库中搜索详细信息"""
                try:
                    results = agent_self.rag_engine.search(query, category="products", k=3)
                    if not results:
                        return "未找到相关产品文档"
                    return "\n\n".join([f"[{doc.metadata.get('source', '知识库')}] {doc.page_content}" for doc in results])
                except Exception as e:
                    return f"检索失败: {str(e)}"

            tools.append(search_product_docs)

        self.system_prompt = """你是华智精密制造的产品顾问。你的职责是：
1. 根据访客需求介绍合适的产品，突出技术优势和应用场景
2. 使用 search_product_catalog 工具查找产品目录
3. 使用 compare_products 工具进行产品对比
4. 使用 get_solution_recommendation 根据场景推荐方案
5. 如有产品文档，使用 search_product_docs 查找详细规格

回复要求：
- 专业且有说服力，面向潜在客户
- 突出产品的认证、品质和应用场景
- 主动推荐适合的方案
- 回复末尾标注信息来源"""

        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        messages = [{"role": "user", "content": message}]
        result = self.agent.invoke({"messages": messages})

        if result["messages"]:
            response = result["messages"][-1].content
            sources = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        sources.append({
                            "type": "tool",
                            "tool": call["name"],
                            "query": str(call.get("args", {}))
                        })
            return {"response": response, "sources": sources}

        return {"response": "抱歉，产品信息暂时无法查询，请稍后再试。", "sources": []}

    def handle_stream(self, message: str, chat_history: List = None):
        try:
            # 查询重写
            search_query = message
            if self.query_processor:
                try:
                    processed = self.query_processor.process(message)
                    search_query = processed["rewritten_query"]
                except Exception:
                    pass

            prompt = self.system_prompt + "\n\n"
            context, sources = _build_rag_context(self.rag_engine, search_query, category="products", k=3)
            if context:
                prompt += f"参考信息：\n{context}\n\n"
            prompt += f"用户问题：{message}"

            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，处理您的请求时出现问题。请稍后再试。"


class TechCapabilityAgent:
    """技术实力代理（集成 RAG）"""

    def __init__(self, rag_engine=None, query_processor=None):
        self.llm = model
        self.rag_engine = rag_engine
        self.query_processor = query_processor

        tools = [get_certification_info]

        if self.rag_engine:
            agent_self = self

            @tool
            def search_tech_docs(query: str) -> str:
                """从技术文档中搜索工艺、认证、质量管控相关信息"""
                try:
                    results = agent_self.rag_engine.search(query, category="capabilities", k=3)
                    if not results:
                        results = agent_self.rag_engine.search(query, category="technical", k=3)
                    if not results:
                        return "未找到相关技术文档"
                    return "\n\n".join([f"[{doc.metadata.get('source', '知识库')}] {doc.page_content}" for doc in results])
                except Exception as e:
                    return f"检索失败: {str(e)}"

            tools.append(search_tech_docs)

        self.system_prompt = """你是华智精密制造的技术实力展示顾问。你的职责是：
1. 展示公司的技术实力、生产工艺和质量管控体系
2. 使用 get_certification_info 查询认证资质
3. 如有技术文档，使用 search_tech_docs 查找详细信息
4. 用数据和案例说话，增强说服力

公司核心数据：
- 成立于2008年，16年精密制造经验
- 1200+专业团队，35000平方米现代化厂房
- 年产能5000万件，年产值8.5亿元
- ISO9001/IATF16949/CE/RoHS认证齐全
- 全自动化生产线，7天快速打样

回复要求：
- 用具体数据和认证增强说服力
- 展现专业性和可靠性
- 适当提及行业案例和客户见证"""

        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        messages = [{"role": "user", "content": message}]
        result = self.agent.invoke({"messages": messages})

        if result["messages"]:
            sources = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        sources.append({"type": "tool", "tool": call["name"], "query": str(call.get("args", {}))})
            return {"response": result["messages"][-1].content, "sources": sources}

        return {"response": "抱歉，技术实力信息暂时无法查询。", "sources": []}

    def handle_stream(self, message: str, chat_history: List = None):
        try:
            search_query = message
            if self.query_processor:
                try:
                    processed = self.query_processor.process(message)
                    search_query = processed["rewritten_query"]
                except Exception:
                    pass

            prompt = self.system_prompt + "\n\n"
            context, _ = _build_rag_context(self.rag_engine, search_query, category="capabilities", k=3)
            if context:
                prompt += f"参考信息：\n{context}\n\n"
            prompt += f"用户问题：{message}"

            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，处理您的请求时出现问题。"


class CompanyOverviewAgent:
    """公司概况代理"""

    def __init__(self, rag_engine=None, query_processor=None):
        self.llm = model
        self.rag_engine = rag_engine
        self.query_processor = query_processor

        self.system_prompt = f"""你是华智精密制造的公司介绍顾问。向访客介绍公司概况。

公司基本信息：
{json.dumps(MOCK_COMPANY, ensure_ascii=False, indent=2)}

回复要求：
- 热情专业，展现企业形象
- 用发展历程和里程碑展示公司实力
- 突出规模、资质和行业地位
- 适当引导访客了解产品或合作方式"""

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        prompt = self.system_prompt + f"\n\n用户问题：{message}"

        if self.rag_engine:
            context, sources = _build_rag_context(self.rag_engine, message, category="company", k=2)
            if context:
                prompt = self.system_prompt + f"\n\n参考信息：\n{context}\n\n用户问题：{message}"
            else:
                sources = []
        else:
            sources = []

        response = self.llm.invoke(prompt)
        return {"response": response.content, "sources": sources}

    def handle_stream(self, message: str, chat_history: List = None):
        try:
            prompt = self.system_prompt + "\n\n"

            if self.rag_engine:
                context, _ = _build_rag_context(self.rag_engine, message, category="company", k=2)
                if context:
                    prompt += f"参考信息：\n{context}\n\n"

            prompt += f"用户问题：{message}"

            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，处理您的请求时出现问题。"


class PartnershipAgent:
    """合作咨询代理"""

    def __init__(self, rag_engine=None, query_processor=None):
        self.llm = model
        self.rag_engine = rag_engine
        self.query_processor = query_processor

        self.system_prompt = f"""你是华智精密制造的商务合作顾问。向潜在合作伙伴介绍合作方式。

合作模式信息：
{json.dumps(MOCK_PARTNERSHIP, ensure_ascii=False, indent=2)}

回复要求：
- 专业、热情、有说服力
- 根据访客需求推荐合适的合作模式（OEM/ODM/深度定制）
- 清晰说明合作流程、起订量、交付周期
- 主动邀请进一步商务接洽
- 提供联系方式：商务邮箱 business@huazhi-mfg.com，热线 400-888-7688"""

    def handle(self, message: str, chat_history: List = None) -> Dict[str, Any]:
        prompt = self.system_prompt + f"\n\n用户问题：{message}"
        response = self.llm.invoke(prompt)
        return {"response": response.content, "sources": []}

    def handle_stream(self, message: str, chat_history: List = None):
        try:
            prompt = self.system_prompt + f"\n\n用户问题：{message}"
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] 流式输出错误: {e}")
            yield "抱歉，处理您的请求时出现问题。"


class QualityChecker:
    """质量检查器"""

    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是企业问答质量检查专家。评估 AI 回复的质量。

评估维度：
1. 相关性（0-25分）：是否针对访客问题
2. 完整性（0-25分）：是否提供了足够信息
3. 专业性（0-25分）：语言是否专业、有说服力
4. 有用性（0-25分）：是否有助于访客了解公司/产品

返回格式（JSON）：
{{"total_score": 0-100, "needs_escalation": true/false, "reason": "评估说明"}}

只返回JSON。"""),
            ("human", """访客问题：{user_message}
AI回复：{agent_response}

请评估：""")
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({"user_message": user_message, "agent_response": agent_response})
        default = {"total_score": 70, "needs_escalation": False, "reason": "评估完成"}
        return safe_parse_json(result, default)


# ==================== 系统编排器 ====================

class EnterpriseQASystem:
    """企业智能问答系统 — LangGraph 工作流编排"""

    def __init__(self, rag_engine=None, enable_cache: bool = True):
        """
        初始化企业问答系统

        Args:
            rag_engine: RAG 引擎实例
            enable_cache: 是否启用响应缓存
        """
        # 查询处理器和重排序器（惰性初始化）
        self.query_processor = None
        self.reranker = None

        try:
            from .query_processor import create_query_processor
            self.query_processor = create_query_processor(enable_rewrite=True, enable_multi_query=False)
        except Exception as e:
            print(f"[WARN] 查询处理器初始化失败: {e}")

        try:
            from .reranker import create_reranker
            self.reranker = create_reranker(enable_llm_rerank=False, top_k=5)
        except Exception as e:
            print(f"[WARN] 重排序器初始化失败: {e}")

        # 初始化代理
        self.classifier = IntentClassifier()
        self.product_agent = ProductInfoAgent(rag_engine, self.query_processor)
        self.tech_agent = TechCapabilityAgent(rag_engine, self.query_processor)
        self.company_agent = CompanyOverviewAgent(rag_engine, self.query_processor)
        self.partnership_agent = PartnershipAgent(rag_engine, self.query_processor)
        self.quality_checker = QualityChecker()

        # 缓存
        self.enable_cache = enable_cache
        self._cache = default_cache if enable_cache else None

        # 构建 LangGraph 工作流
        self.graph = self._build_graph()
        print("[EnterpriseQA] 企业智能问答系统初始化完成")

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""

        def classify_intent(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[INTENT] 分析访客意图...")
            result = self.classifier.classify(state["user_message"])
            state["intent"] = result.get("intent", "company_overview")
            state["confidence"] = result.get("confidence", 0.5)
            print(f"   意图: {state['intent']} (置信度: {state['confidence']:.2f})")
            return state

        def route_to_agent(state: EnterpriseQAState) -> Literal[
            "product_info", "tech_capability", "company_overview", "partnership", "escalate"
        ]:
            intent = state["intent"]
            confidence = state["confidence"]

            if confidence < 0.6:
                return "escalate"

            routing = {
                "product_info": "product_info",
                "tech_capability": "tech_capability",
                "company_overview": "company_overview",
                "partnership": "partnership",
            }
            return routing.get(intent, "escalate")

        def product_info_handler(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[PRODUCT] 产品咨询代理处理中...")
            result = self.product_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def tech_capability_handler(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[TECH] 技术实力代理处理中...")
            result = self.tech_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def company_overview_handler(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[COMPANY] 公司概况代理处理中...")
            result = self.company_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def partnership_handler(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[PARTNER] 合作咨询代理处理中...")
            result = self.partnership_agent.handle(state["user_message"])
            state["agent_response"] = result["response"]
            state["sources"] = result.get("sources", [])
            return state

        def escalate_handler(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[ESCALATE] 转接人工...")
            state["needs_escalation"] = True
            state["escalation_reason"] = "意图识别置信度低或需要人工接待"
            state["agent_response"] = """感谢您的咨询！您的问题需要我们的专业团队为您详细解答。

您可以通过以下方式联系我们：
1. 商务热线：400-888-7688
2. 商务邮箱：business@huazhi-mfg.com
3. 工作日 9:00-18:00，响应更及时

我们的专业团队将在第一时间为您提供详细的解决方案。"""
            state["sources"] = []
            return state

        def quality_check(state: EnterpriseQAState) -> EnterpriseQAState:
            print("[QA] 执行质量检查...")
            result = self.quality_checker.check(
                state["user_message"],
                state["agent_response"]
            )
            state["quality_score"] = result.get("total_score", 0) / 100
            if result.get("needs_escalation", False) or state["quality_score"] < 0.5:
                state["needs_escalation"] = True
                state["escalation_reason"] = result.get("reason", "质量检查未通过")
            print(f"   质量评分: {state['quality_score']:.2f}")
            return state

        def should_escalate(state: EnterpriseQAState) -> Literal["escalate_final", "respond"]:
            if state.get("needs_escalation", False):
                return "escalate_final"
            return "respond"

        def final_escalate(state: EnterpriseQAState) -> EnterpriseQAState:
            original = state["agent_response"]
            state["agent_response"] = f"""{original}

---
温馨提示：如需更详细的解答，欢迎联系我们的专业团队：
- 商务热线：400-888-7688
- 邮箱：business@huazhi-mfg.com"""
            return state

        def respond(state: EnterpriseQAState) -> EnterpriseQAState:
            return state

        # 构建图
        graph = StateGraph(EnterpriseQAState)

        # 添加节点
        graph.add_node("classify", classify_intent)
        graph.add_node("product_info", product_info_handler)
        graph.add_node("tech_capability", tech_capability_handler)
        graph.add_node("company_overview", company_overview_handler)
        graph.add_node("partnership", partnership_handler)
        graph.add_node("escalate", escalate_handler)
        graph.add_node("quality_check", quality_check)
        graph.add_node("escalate_final", final_escalate)
        graph.add_node("respond", respond)

        # 添加边
        graph.add_edge(START, "classify")

        graph.add_conditional_edges(
            "classify",
            route_to_agent,
            {
                "product_info": "product_info",
                "tech_capability": "tech_capability",
                "company_overview": "company_overview",
                "partnership": "partnership",
                "escalate": "escalate"
            }
        )

        # 代理 → 质量检查
        graph.add_edge("product_info", "quality_check")
        graph.add_edge("tech_capability", "quality_check")
        graph.add_edge("company_overview", "quality_check")
        graph.add_edge("partnership", "quality_check")
        graph.add_edge("escalate", END)

        # 质量检查 → 最终路由
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
        print(f"[Q&A] 访客: {message}")
        print('='*60)

        if self.enable_cache and self._cache:
            cached_result = self._cache.get(message, agent_type="enterprise_qa")
            if cached_result is not None:
                print("[CACHE] 命中缓存")
                return cached_result

        initial_state = EnterpriseQAState(
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

        response = {
            "response": result["agent_response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "quality_score": result["quality_score"],
            "escalated": result["needs_escalation"],
            "sources": result.get("sources", [])
        }

        if self.enable_cache and self._cache:
            self._cache.set(message, response, agent_type="enterprise_qa")

        return response

    def handle_message_stream(self, message: str, chat_history: List[Dict] = None):
        """处理用户消息（流式输出）"""
        print(f"\n{'='*60}")
        print(f"[Q&A STREAM] 访客: {message}")
        print('='*60)

        # 1. 意图识别
        result = self.classifier.classify(message)
        intent = result.get("intent", "company_overview")
        confidence = result.get("confidence", 0.0)

        yield {
            "type": "intent",
            "intent": intent,
            "confidence": confidence,
            "done": False
        }

        # 2. 路由到对应代理流式输出
        try:
            agent_map = {
                "product_info": self.product_agent,
                "tech_capability": self.tech_agent,
                "company_overview": self.company_agent,
                "partnership": self.partnership_agent,
            }

            if intent in agent_map:
                for chunk in agent_map[intent].handle_stream(message):
                    yield {"type": "content", "content": chunk, "done": False}
            else:
                escalate_msg = """感谢您的咨询！您的问题需要我们的专业团队为您详细解答。

联系方式：
1. 商务热线：400-888-7688
2. 商务邮箱：business@huazhi-mfg.com
3. 工作日 9:00-18:00"""
                for i in range(0, len(escalate_msg), 15):
                    yield {"type": "content", "content": escalate_msg[i:i+15], "done": False}

            # 3. 获取完整元数据
            final_result = self.graph.invoke(
                EnterpriseQAState(
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

            sources = final_result.get("sources", [])
            serializable_sources = []
            for s in sources:
                if isinstance(s, dict):
                    serializable_sources.append(s)
                else:
                    serializable_sources.append({"type": str(type(s).__name__), "content": str(s)})

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
            yield {"type": "content", "content": "抱歉，处理您的请求时出现了错误。请稍后再试。", "done": False}
            yield {"type": "error", "error": str(e), "done": True}


# ==================== 向后兼容别名 ====================

# 保持与 backend/main.py 的兼容性
CustomerServiceSystem = EnterpriseQASystem
