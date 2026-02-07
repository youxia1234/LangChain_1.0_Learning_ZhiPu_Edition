"""
模块 21：混合模态
学习如何处理文本、图像等多种模态的输入

⚠️ 重要提示：
1. 本模块需要支持视觉的模型（使用智谱 AI 的 glm-4v 模型）
2. 请在 images/ 目录下放置你自己的测试图片

使用前准备：
1. 在 .env 中设置 ZHIPUAI_API_KEY
2. 在 images/ 目录下放置测试图片
"""

import os
import sys
import base64
from pathlib import Path
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "图像处理需要使用智谱 AI 的视觉模型（glm-4v）\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI 的视觉模型）
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="glm-4v",  # 智谱 AI 的视觉模型
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 图片目录
IMAGES_DIR = Path(__file__).parent / "images"

# ============================================================
# 辅助函数
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """将本地图像编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def get_mime_type(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return mime_types.get(ext, "image/jpeg")

def create_image_content(image_path: str) -> dict:
    """创建图像内容块"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
    }

def check_image_exists(filename: str) -> str:
    """检查图片是否存在"""
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        print(f"⚠️ 图片不存在: {image_path}")
        return None
    return str(image_path)

# ============================================================
# 示例 1：文本 + 图像混合输入
# ============================================================

def example_1_text_and_image():
    """
    处理文本和图像的混合输入
    """
    print("\n" + "=" * 60)
    print("示例 1：文本 + 图像混合输入")
    print("=" * 60)
    
    image_path = check_image_exists("chart.png")
    if not image_path:
        print("请在 images/ 目录下放置 chart.png（图表图片）")
        print("跳过此示例")
        return None
    
    # 创建混合内容消息
    content = [
        {"type": "text", "text": """以下是我们公司的销售数据：

**2024年第一季度销售报告**
- 1月: 150万
- 2月: 180万  
- 3月: 220万

请结合图表分析：
1. 数据趋势如何？
2. 与图表显示的趋势是否一致？
3. 你有什么建议？"""},
        create_image_content(image_path)
    ]
    
    message = HumanMessage(content=content)
    
    print("📊 发送文本数据 + 图表图片...")
    
    response = model.invoke([message])
    
    print("\n🤖 分析结果：")
    print(response.content)
    
    return response.content

# ============================================================
# 示例 2：多图像对比分析
# ============================================================

def example_2_multi_image():
    """
    对比多张图片
    """
    print("\n" + "=" * 60)
    print("示例 2：多图像对比分析")
    print("=" * 60)
    
    # 检查图片
    image1_path = check_image_exists("image1.jpg")
    image2_path = check_image_exists("image2.jpg")
    
    if not image1_path or not image2_path:
        print("请在 images/ 目录下放置 image1.jpg 和 image2.jpg")
        print("跳过此示例")
        return None
    
    content = [
        {"type": "text", "text": "请对比这两张图片，说明它们的相同点和不同点。"},
        create_image_content(image1_path),
        create_image_content(image2_path)
    ]
    
    message = HumanMessage(content=content)
    
    print("📷 对比两张图片...")
    
    response = model.invoke([message])
    
    print("\n🔍 对比结果：")
    print(response.content)
    
    return response.content

# ============================================================
# 示例 3：使用 LangGraph 处理混合模态
# ============================================================

class MultimodalState(TypedDict):
    """混合模态状态"""
    text_input: str
    image_paths: List[str]
    analysis_result: Optional[str]
    summary: Optional[str]

def example_3_langgraph_multimodal():
    """
    使用 LangGraph 构建混合模态处理流程
    """
    print("\n" + "=" * 60)
    print("示例 3：LangGraph 混合模态处理")
    print("=" * 60)
    
    # 检查图片
    image_path = check_image_exists("sample.jpg")
    if not image_path:
        print("请在 images/ 目录下放置 sample.jpg")
        print("跳过此示例")
        return None
    
    # 定义节点函数
    def analyze_content(state: MultimodalState) -> MultimodalState:
        """分析混合内容"""
        print("📝 正在分析内容...")
        
        content = [{"type": "text", "text": state["text_input"]}]
        
        for img_path in state["image_paths"]:
            if os.path.exists(img_path):
                content.append(create_image_content(img_path))
        
        message = HumanMessage(content=content)
        response = model.invoke([message])
        
        state["analysis_result"] = response.content
        return state
    
    def summarize(state: MultimodalState) -> MultimodalState:
        """总结分析结果"""
        print("📋 正在生成总结...")
        
        message = HumanMessage(
            content=f"请用3句话总结以下分析：\n\n{state['analysis_result']}"
        )
        response = model.invoke([message])
        
        state["summary"] = response.content
        return state
    
    # 构建图
    graph = StateGraph(MultimodalState)
    
    graph.add_node("analyze", analyze_content)
    graph.add_node("summarize", summarize)
    
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", END)
    
    workflow = graph.compile()
    
    # 运行
    initial_state = {
        "text_input": "请详细描述这张图片，包括主要内容、色彩和氛围。",
        "image_paths": [image_path],
        "analysis_result": None,
        "summary": None
    }
    
    result = workflow.invoke(initial_state)
    
    print("\n📊 详细分析：")
    print(result["analysis_result"])
    print("\n📌 总结：")
    print(result["summary"])
    
    return result

# ============================================================
# 示例 4：交互式图像问答
# ============================================================

def example_4_interactive_qa():
    """
    基于图像的交互式问答
    """
    print("\n" + "=" * 60)
    print("示例 4：交互式图像问答（演示）")
    print("=" * 60)
    
    image_path = check_image_exists("sample.jpg")
    if not image_path:
        print("请在 images/ 目录下放置 sample.jpg")
        print("跳过此示例")
        return None
    
    # 模拟问答流程
    questions = [
        "这张图片展示了什么？",
        "图片中有哪些颜色？",
        "你觉得这张图片是在什么场景下拍摄的？"
    ]
    
    messages = []
    
    # 首先发送图片
    initial_content = [
        {"type": "text", "text": "我将基于这张图片问你一些问题。"},
        create_image_content(image_path)
    ]
    messages.append(HumanMessage(content=initial_content))
    
    print(f"📷 已加载图片: {image_path}\n")
    
    for q in questions:
        print(f"❓ 问题: {q}")
        
        messages.append(HumanMessage(content=q))
        response = model.invoke(messages)
        messages.append(response)
        
        print(f"💬 回答: {response.content}\n")
    
    print("💡 提示: 实际应用中可以使用 input() 实现真正的交互式问答")
    
    return messages

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("混合模态处理教程")
    print("=" * 60)

    print("""
⚠️ 使用前请确保：
1. 已在 .env 中设置 ZHIPUAI_API_KEY
2. 已在 images/ 目录下放置测试图片:
   - sample.jpg: 通用测试图片
   - chart.png: 图表图片
   - image1.jpg, image2.jpg: 用于对比的图片

如果没有准备图片，相关示例将被跳过。
""")
    
    # 创建图片目录
    IMAGES_DIR.mkdir(exist_ok=True)
    
    # 运行示例
    example_1_text_and_image()
    example_2_multi_image()
    example_3_langgraph_multimodal()
    example_4_interactive_qa()
    
    print("\n" + "=" * 60)
    print("✅ 教程运行完成！")
    print("=" * 60)
    print("""
💡 学习要点：
1. 混合模态消息的构建方式
2. 多图像输入的处理
3. 使用 LangGraph 处理混合模态流程
4. 交互式图像问答的实现

📝 下一步：
- 尝试使用自己的图片
- 修改提示词观察效果变化
- 结合 RAG 实现更复杂的应用
""")
