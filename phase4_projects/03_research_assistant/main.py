"""
项目三：智能研究助手

本项目构建一个完整的研究助手系统，能够进行多源信息收集、
文献分析、知识整合和研究报告生成。

核心特性：
- 多源信息收集（模拟网络搜索、学术数据库）
- 文档分析和关键信息提取
- 知识图谱构建和关联分析
- 自动化研究报告生成
- 引用管理和来源追踪

Author: LangChain 学习项目
Version: 1.0
"""

import os
import sys
import json
from typing import TypedDict, Literal, Annotated, Optional
from datetime import datetime
from dotenv import load_dotenv

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

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
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON 解析失败: {e}")
        return default



# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ==================== 数据模型定义 ====================

class SearchResult(BaseModel):
    """搜索结果"""
    title: str
    source: str
    url: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    publish_date: Optional[str] = None

class ResearchFinding(BaseModel):
    """研究发现"""
    topic: str
    key_points: list[str]
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]

class ResearchOutline(BaseModel):
    """研究大纲"""
    title: str
    abstract: str
    sections: list[str]
    key_questions: list[str]
    methodology: str

class Citation(BaseModel):
    """引用"""
    id: str
    authors: list[str]
    title: str
    source: str
    year: int
    url: Optional[str] = None

class ResearchReport(BaseModel):
    """研究报告"""
    title: str
    executive_summary: str
    introduction: str
    methodology: str
    findings: list[str]
    analysis: str
    conclusions: list[str]
    recommendations: list[str]
    citations: list[Citation]
    generated_at: str

# ==================== 状态定义 ====================

class ResearchState(TypedDict):
    """研究助手状态"""
    # 消息历史
    messages: Annotated[list, add_messages]
    
    # 研究主题
    research_topic: str
    research_questions: list[str]
    
    # 收集的数据
    search_results: list[dict]
    analyzed_sources: list[dict]
    
    # 研究进展
    outline: dict
    findings: list[dict]
    
    # 报告
    draft_sections: dict
    final_report: str
    citations: list[dict]
    
    # 状态追踪
    current_phase: str
    iteration_count: int
    quality_score: float

# ==================== 模拟数据源 ====================

# 模拟学术数据库
ACADEMIC_DATABASE = {
    "人工智能": [
        {
            "title": "深度学习在自然语言处理中的应用综述",
            "authors": ["张明", "李华"],
            "source": "计算机学报",
            "year": 2024,
            "snippet": "本文综述了深度学习技术在NLP领域的最新进展，包括Transformer架构、预训练模型和大语言模型的发展。",
            "url": "https://example.com/paper1"
        },
        {
            "title": "大语言模型的涌现能力研究",
            "authors": ["王强", "赵丽"],
            "source": "人工智能研究",
            "year": 2024,
            "snippet": "研究发现，当模型规模超过一定阈值时，会出现思维链推理、上下文学习等涌现能力。",
            "url": "https://example.com/paper2"
        },
        {
            "title": "AI Agent系统的设计与实现",
            "authors": ["刘伟", "陈静"],
            "source": "软件工程学报",
            "year": 2024,
            "snippet": "提出了一种基于LLM的智能体架构，支持工具调用、规划和多智能体协作。",
            "url": "https://example.com/paper3"
        }
    ],
    "气候变化": [
        {
            "title": "全球气候变化对农业的影响评估",
            "authors": ["孙涛", "周明"],
            "source": "环境科学学报",
            "year": 2024,
            "snippet": "研究表明，气候变化导致全球主要农作物产量波动加剧，极端天气事件频发。",
            "url": "https://example.com/paper4"
        },
        {
            "title": "碳中和路径与技术创新",
            "authors": ["吴芳", "郑强"],
            "source": "能源研究",
            "year": 2024,
            "snippet": "分析了实现碳中和目标的技术路径，包括可再生能源、碳捕集和储能技术。",
            "url": "https://example.com/paper5"
        }
    ],
    "量子计算": [
        {
            "title": "量子计算机的发展现状与挑战",
            "authors": ["黄伟", "林小明"],
            "source": "物理学报",
            "year": 2024,
            "snippet": "综述了超导、离子阱和光量子等技术路线的最新进展，讨论了量子纠错和扩展性挑战。",
            "url": "https://example.com/paper6"
        }
    ]
}

# 模拟网络搜索结果
WEB_SEARCH_RESULTS = {
    "人工智能": [
        {
            "title": "OpenAI发布GPT-5：AI能力再次飞跃",
            "source": "科技新闻网",
            "url": "https://news.example.com/ai1",
            "snippet": "最新发布的GPT-5在推理能力和多模态理解上取得重大突破...",
            "date": "2024-12"
        },
        {
            "title": "企业AI应用调查报告：85%企业已部署AI系统",
            "source": "商业周刊",
            "url": "https://news.example.com/ai2",
            "snippet": "调查显示，大多数企业已在客服、分析和自动化领域应用AI技术...",
            "date": "2024-11"
        }
    ],
    "气候变化": [
        {
            "title": "COP29峰会达成新气候协议",
            "source": "环球时报",
            "url": "https://news.example.com/climate1",
            "snippet": "各国承诺加速减排进程，发达国家将提供更多气候融资...",
            "date": "2024-11"
        }
    ],
    "量子计算": [
        {
            "title": "Google实现量子霸权2.0：1分钟完成超算1万年任务",
            "source": "科学日报",
            "url": "https://news.example.com/quantum1",
            "snippet": "新型量子处理器在特定计算任务上展现出压倒性优势...",
            "date": "2024-10"
        }
    ]
}

# ==================== 工具函数 ====================

def search_academic_database(topic: str, max_results: int = 5) -> list[dict]:
    """搜索学术数据库"""
    results = []
    
    for key, papers in ACADEMIC_DATABASE.items():
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            for paper in papers[:max_results]:
                results.append({
                    **paper,
                    "type": "academic",
                    "relevance_score": 0.9
                })
    
    # 如果没有精确匹配，返回部分相关结果
    if not results:
        for papers in ACADEMIC_DATABASE.values():
            results.extend(papers[:2])
            if len(results) >= max_results:
                break
        for r in results:
            r["relevance_score"] = 0.5
    
    return results[:max_results]

def search_web(topic: str, max_results: int = 5) -> list[dict]:
    """模拟网络搜索"""
    results = []
    
    for key, items in WEB_SEARCH_RESULTS.items():
        if topic.lower() in key.lower() or key.lower() in topic.lower():
            for item in items[:max_results]:
                results.append({
                    **item,
                    "type": "web",
                    "relevance_score": 0.8
                })
    
    if not results:
        for items in WEB_SEARCH_RESULTS.values():
            results.extend(items[:2])
            if len(results) >= max_results:
                break
        for r in results:
            r["relevance_score"] = 0.4
    
    return results[:max_results]

def format_citation(source: dict, citation_id: str) -> Citation:
    """格式化引用"""
    return Citation(
        id=citation_id,
        authors=source.get("authors", ["Unknown"]),
        title=source.get("title", "Untitled"),
        source=source.get("source", "Unknown"),
        year=source.get("year", 2024),
        url=source.get("url")
    )

# ==================== 智能体节点 ====================

def create_research_assistant():
    """创建研究助手系统"""
    
    # ---- 研究规划节点 ----
    def planning_node(state: ResearchState) -> dict:
        """规划研究方向和大纲"""
        print("\n" + "="*50)
        print("📋 研究规划阶段...")
        
        topic = state["research_topic"]
        
        # 修改prompt，明确要求JSON格式
        planning_prompt = f"""你是一位资深研究员。请为以下研究主题制定研究计划。

研究主题：{topic}

请严格按照以下JSON格式返回（不要添加任何其他内容）：
{{
    "title": "研究标题",
    "abstract": "摘要（100字以内）",
    "sections": ["章节1", "章节2", "章节3", "章节4"],
    "key_questions": ["问题1", "问题2", "问题3"],
    "methodology": "研究方法论描述"
}}

只返回JSON，不要其他文字。"""
        
        response = model.invoke([HumanMessage(content=planning_prompt)])
        
        # 使用安全的 JSON 解析，提供默认值
        default_outline = {
            "title": f"{topic}研究",
            "abstract": f"本研究探讨{topic}的相关问题。",
            "sections": ["引言", "文献综述", "研究方法", "结果分析", "结论"],
            "key_questions": [f"{topic}的现状如何？", f"{topic}的发展趋势是什么？", f"{topic}面临哪些挑战？"],
            "methodology": "文献研究与案例分析相结合"
        }
        
        outline_data = safe_parse_json(response.content, default_outline)
        
        # 确保所有必需字段都存在
        for key in default_outline:
            if key not in outline_data:
                outline_data[key] = default_outline[key]
        
        print(f"   标题: {outline_data.get('title', 'N/A')}")
        print(f"   章节数: {len(outline_data.get('sections', []))}")
        print(f"   研究问题数: {len(outline_data.get('key_questions', []))}")
        
        return {
            "outline": outline_data,
            "research_questions": outline_data.get("key_questions", []),
            "current_phase": "information_gathering",
            "messages": [AIMessage(content=f"研究计划已制定：{outline_data.get('title', topic)}")]
        }
    
    # ---- 信息收集节点 ----
    def information_gathering_node(state: ResearchState) -> dict:
        """收集相关信息"""
        print("\n" + "-"*50)
        print("🔍 信息收集阶段...")
        
        topic = state["research_topic"]
        
        # 搜索学术数据库
        print("   搜索学术数据库...")
        academic_results = search_academic_database(topic)
        print(f"   找到 {len(academic_results)} 篇学术文献")
        
        # 搜索网络
        print("   搜索网络资源...")
        web_results = search_web(topic)
        print(f"   找到 {len(web_results)} 条网络结果")
        
        all_results = academic_results + web_results
        
        # 显示搜索结果
        print("\n   收集到的资料：")
        for i, result in enumerate(all_results[:5], 1):
            print(f"   {i}. [{result['type']}] {result['title']}")
        
        return {
            "search_results": all_results,
            "current_phase": "analysis",
            "messages": [AIMessage(content=f"已收集 {len(all_results)} 条相关资料")]
        }
    
    # ---- 信息分析节点 ----
    def analysis_node(state: ResearchState) -> dict:
        """分析收集的信息"""
        print("\n" + "-"*50)
        print("📊 信息分析阶段...")
        
        topic = state["research_topic"]
        search_results = state.get("search_results", [])
        research_questions = state.get("research_questions", [])
        
        # 整理资料摘要
        sources_summary = "\n".join([
            f"- {r['title']}: {r.get('snippet', '')}"
            for r in search_results[:8]
        ])
        
        analysis_prompt = f"""基于以下资料，对研究主题进行深入分析：

研究主题：{topic}

核心问题：
{chr(10).join(f'- {q}' for q in research_questions)}

收集的资料：
{sources_summary}

请提供：
1. 对每个核心问题的初步回答
2. 资料中的关键发现
3. 不同观点的比较
4. 信息空白和需要进一步研究的方向

用JSON格式输出，包含 key_findings, analysis_points, information_gaps 三个字段。"""
        
        response = model.invoke([HumanMessage(content=analysis_prompt)])
        
        # 解析分析结果（简化处理）
        findings = [
            {
                "topic": topic,
                "key_points": [
                    "多项研究表明该领域正在快速发展",
                    "存在多种技术路线和方法论",
                    "实际应用案例增加"
                ],
                "confidence": 0.85,
                "sources": [r["title"] for r in search_results[:3]]
            }
        ]
        
        analyzed_sources = []
        for i, result in enumerate(search_results[:6]):
            analyzed_sources.append({
                "id": f"src_{i+1}",
                "title": result["title"],
                "key_takeaways": result.get("snippet", "")[:100],
                "relevance": result.get("relevance_score", 0.5)
            })
        
        print(f"   分析了 {len(analyzed_sources)} 个来源")
        print(f"   提取了 {len(findings)} 组关键发现")
        
        return {
            "findings": findings,
            "analyzed_sources": analyzed_sources,
            "current_phase": "synthesis",
            "messages": [AIMessage(content=response.content)]
        }
    
    # ---- 知识综合节点 ----
    def synthesis_node(state: ResearchState) -> dict:
        """综合知识，生成报告初稿"""
        print("\n" + "-"*50)
        print("🔮 知识综合阶段...")
        
        topic = state["research_topic"]
        outline = state.get("outline", {})
        findings = state.get("findings", [])
        analyzed_sources = state.get("analyzed_sources", [])
        
        # 生成各章节内容
        sections = outline.get("sections", ["引言", "背景", "发现", "结论"])
        
        synthesis_prompt = f"""基于研究大纲和分析结果，为以下研究报告生成内容：

研究主题：{topic}

研究大纲：
- 标题: {outline.get('title', topic)}
- 摘要: {outline.get('abstract', '')}
- 章节: {', '.join(sections)}

关键发现：
{json.dumps(findings, ensure_ascii=False, indent=2)}

参考来源：
{chr(10).join(f"- [{s['id']}] {s['title']}" for s in analyzed_sources[:5])}

请为每个章节生成200-300字的内容，使用学术写作风格。
在引用观点时，标注来源ID，如 [src_1]。"""
        
        response = model.invoke([HumanMessage(content=synthesis_prompt)])
        
        # 构建章节内容（简化）
        draft_sections = {
            "introduction": "（研究背景和目的）",
            "methodology": outline.get("methodology", "文献综述与分析"),
            "findings": response.content,
            "conclusion": "（研究结论）"
        }
        
        print(f"   生成了 {len(draft_sections)} 个章节")
        
        return {
            "draft_sections": draft_sections,
            "current_phase": "report_generation",
            "messages": [AIMessage(content="报告初稿已完成")]
        }
    
    # ---- 报告生成节点 ----
    def report_generation_node(state: ResearchState) -> dict:
        """生成最终研究报告"""
        print("\n" + "-"*50)
        print("📝 报告生成阶段...")
        
        topic = state["research_topic"]
        outline = state.get("outline", {})
        draft_sections = state.get("draft_sections", {})
        search_results = state.get("search_results", [])
        
        # 生成引用列表
        citations = []
        for i, result in enumerate(search_results[:6]):
            citation = {
                "id": f"[{i+1}]",
                "authors": result.get("authors", ["Unknown"]),
                "title": result.get("title", ""),
                "source": result.get("source", ""),
                "year": result.get("year", 2024),
                "url": result.get("url", "")
            }
            citations.append(citation)
        
        # 生成完整报告
        report_prompt = f"""请将以下研究内容整合为一份完整的研究报告：

标题：{outline.get('title', topic + ' 研究报告')}

摘要：{outline.get('abstract', '')}

章节内容：
{json.dumps(draft_sections, ensure_ascii=False, indent=2)}

参考文献（请在报告末尾列出）：
{chr(10).join(f"{c['id']} {', '.join(c['authors'])}. {c['title']}. {c['source']}, {c['year']}." for c in citations)}

请输出格式规范、结构清晰的研究报告，包括：
1. 标题
2. 摘要
3. 目录
4. 正文各章节
5. 结论与建议
6. 参考文献"""
        
        response = model.invoke([HumanMessage(content=report_prompt)])
        
        final_report = response.content
        
        print(f"   报告字数: {len(final_report)}")
        print(f"   引用数量: {len(citations)}")
        
        return {
            "final_report": final_report,
            "citations": citations,
            "current_phase": "quality_check",
            "messages": [AIMessage(content="研究报告已生成")]
        }
    
    # ---- 质量检查节点 ----
    def quality_check_node(state: ResearchState) -> dict:
        """检查报告质量"""
        print("\n" + "-"*50)
        print("✅ 质量检查阶段...")
        
        final_report = state.get("final_report", "")
        citations = state.get("citations", [])
        
        # 评估质量
        quality_prompt = f"""请评估以下研究报告的质量（0-10分）：

报告摘要（前500字）：
{final_report[:500]}...

引用数量：{len(citations)}

评估维度：
1. 结构完整性
2. 论证逻辑性
3. 引用规范性
4. 语言表达
5. 学术严谨性

请给出总分和改进建议。"""
        
        response = model.invoke([HumanMessage(content=quality_prompt)])
        
        # 简化质量分数计算
        quality_score = 8.0 if len(final_report) > 1000 and len(citations) >= 3 else 6.5
        
        print(f"   质量评分: {quality_score}/10")
        
        return {
            "quality_score": quality_score,
            "current_phase": "completed",
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": [AIMessage(content=f"质量评估完成，得分：{quality_score}/10\n\n{response.content}")]
        }
    
    # ---- 路由函数 ----
    def should_continue(state: ResearchState) -> Literal["continue", "complete"]:
        """判断是否需要继续迭代"""
        quality_score = state.get("quality_score", 0)
        iteration_count = state.get("iteration_count", 0)
        
        if quality_score >= 7.5 or iteration_count >= 2:
            return "complete"
        return "continue"
    
# ==================== 构建图 ====================
    
    graph = StateGraph(ResearchState)
    
    # 添加节点
    graph.add_node("planning", planning_node)
    graph.add_node("information_gathering", information_gathering_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("quality_check", quality_check_node)
    
    # 设置流程
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "information_gathering")
    graph.add_edge("information_gathering", "analysis")
    graph.add_edge("analysis", "synthesis")
    graph.add_edge("synthesis", "report_generation")
    graph.add_edge("report_generation", "quality_check")
    
    # 条件路由：根据质量决定是否重新迭代
    graph.add_conditional_edges(
        "quality_check",
        should_continue,
        {
            "continue": "analysis",  # 重新分析
            "complete": END
        }
    )
    
    # 编译
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    return compiled_graph

# ==================== 运行研究任务 ====================

def run_research(topic: str):
    """运行研究任务"""
    print("\n" + "="*60)
    print("🔬 启动研究任务")
    print("="*60)
    print(f"研究主题: {topic}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建研究助手
    assistant = create_research_assistant()
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content=f"请对以下主题进行深入研究：{topic}")],
        "research_topic": topic,
        "research_questions": [],
        "search_results": [],
        "analyzed_sources": [],
        "outline": {},
        "findings": [],
        "draft_sections": {},
        "final_report": "",
        "citations": [],
        "current_phase": "planning",
        "iteration_count": 0,
        "quality_score": 0.0
    }
    
    # 运行研究流程
    config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
    result = assistant.invoke(initial_state, config)
    
    # 输出结果
    print("\n" + "="*60)
    print("📄 研究报告")
    print("="*60)
    print(result.get("final_report", "报告生成失败"))
    
    print("\n" + "-"*60)
    print("📚 参考文献")
    print("-"*60)
    for citation in result.get("citations", []):
        authors = ", ".join(citation.get("authors", ["Unknown"]))
        print(f"{citation['id']} {authors}. {citation['title']}. {citation['source']}, {citation['year']}.")
    
    print("\n" + "-"*60)
    print("📊 研究统计")
    print("-"*60)
    print(f"  - 收集资料数: {len(result.get('search_results', []))}")
    print(f"  - 分析来源数: {len(result.get('analyzed_sources', []))}")
    print(f"  - 迭代次数: {result.get('iteration_count', 0)}")
    print(f"  - 质量评分: {result.get('quality_score', 0):.1f}/10")
    print(f"  - 报告字数: {len(result.get('final_report', ''))}")
    
    return result

# ==================== 高级功能演示 ====================

def demonstrate_advanced_features():
    """演示高级功能"""
    print("\n" + "="*60)
    print("🚀 高级功能演示")
    print("="*60)
    
        
    # 功能1：多主题比较研究
    print("\n📊 功能1：多主题比较分析")
    print("-"*50)
    
    topics = ["人工智能", "量子计算", "气候变化"]
    comparisons = []
    
    for topic in topics:
        academic = search_academic_database(topic, max_results=2)
        web = search_web(topic, max_results=2)
        comparisons.append({
            "topic": topic,
            "academic_count": len(academic),
            "web_count": len(web),
            "total_sources": len(academic) + len(web)
        })
        print(f"  {topic}: 学术 {len(academic)} 篇, 网络 {len(web)} 条")
    
    # 功能2：研究趋势分析
    print("\n📈 功能2：研究趋势分析")
    print("-"*50)
    
    trend_prompt = """基于以下主题的研究资料数量，分析当前研究热点趋势：
    
- 人工智能：学术文献3篇，新闻报道2条
- 量子计算：学术文献1篇，新闻报道1条
- 气候变化：学术文献2篇，新闻报道1条

请简要分析（100字以内）："""
    
    trend_analysis = model.invoke([HumanMessage(content=trend_prompt)])
    print(f"  趋势分析: {trend_analysis.content}")
    
    # 功能3：智能文献推荐
    print("\n📚 功能3：智能文献推荐")
    print("-"*50)
    
    user_interest = "我对AI Agent的应用很感兴趣"
    
    recommend_prompt = f"""用户兴趣：{user_interest}

可用文献：
1. 深度学习在自然语言处理中的应用综述
2. 大语言模型的涌现能力研究
3. AI Agent系统的设计与实现
4. 全球气候变化对农业的影响评估
5. 量子计算机的发展现状与挑战

请推荐最相关的2篇文献，并简要说明原因（50字以内）。"""
    
    recommendations = model.invoke([HumanMessage(content=recommend_prompt)])
    print(f"  推荐结果:\n{recommendations.content}")

# ==================== 主程序 ====================

def main():
    """主程序"""
    print("="*60)
    print("🔬 智能研究助手系统")
    print("="*60)
    
    # 研究主题列表
    research_topics = [
        "人工智能在医疗诊断中的应用",
        # "量子计算的商业化前景",  # 可选更多主题
        # "碳中和技术发展路径"
    ]
    
    # 运行研究任务
    for topic in research_topics:
        print(f"\n{'#'*60}")
        print(f"# 研究主题: {topic}")
        print(f"{'#'*60}")
        
        result = run_research(topic)
        
        print("\n⏳ 研究完成\n")
    
    # 演示高级功能
    demonstrate_advanced_features()
    
    print("\n" + "="*60)
    print("✅ 所有任务完成！")
    print("="*60)
    
    # 使用说明
    print("\n💡 扩展使用示例:")
    print("-"*60)
    print("""
# 自定义研究主题
result = run_research("区块链在供应链管理中的应用")

# 访问研究结果
print(result["final_report"])  # 完整报告
print(result["citations"])      # 参考文献
print(result["findings"])       # 关键发现

# 导出报告
with open("research_report.md", "w") as f:
    f.write(result["final_report"])
""")

if __name__ == "__main__":
    main()
