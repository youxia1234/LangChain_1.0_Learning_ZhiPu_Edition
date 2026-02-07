"""
模块 19：图像输入
学习如何使用视觉模型处理图像

⚠️ 重要提示：
1. 本模块需要支持视觉的模型（使用智谱 AI 的 glm-4v 模型）
2. 请在 images/ 目录下放置你自己的测试图片

使用前准备：
1. 在 .env 中设置 ZHIPUAI_API_KEY
2. 在 images/ 目录下放置以下图片（或使用你自己的图片）:
   - sample.jpg: 任意测试图片
   - text_image.jpg: 包含文字的图片（用于OCR测试）
   - chart.png: 图表图片（用于图表分析）
"""

import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

def create_image_message(text: str, image_path: str) -> HumanMessage:
    """
    创建包含本地图像的消息
    
    Args:
        text: 文字提示
        image_path: 本地图片路径
    
    Returns:
        HumanMessage 对象
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    
    content = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
        }
    ]
    
    return HumanMessage(content=content)

def check_image_exists(filename: str) -> str:
    """
    检查图片是否存在，返回完整路径
    如果不存在则提示用户
    """
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        print(f"\n⚠️ 图片不存在: {image_path}")
        print(f"请将图片 '{filename}' 放入 images/ 目录")
        print("或者修改代码使用你自己的图片路径\n")
        return None
    return str(image_path)

# ============================================================
# 示例 1：基本图像描述
# ============================================================

def example_1_image_description():
    """
    让模型描述图片内容
    """
    print("\n" + "=" * 60)
    print("示例 1：基本图像描述")
    print("=" * 60)

    # 检查图片是否存在
    image_path = check_image_exists("sample.jpg")
    if not image_path:
        print("跳过此示例")
        return None
    
    message = create_image_message(
        text="请详细描述这张图片中的内容。用中文回复。",
        image_path=image_path
    )
    
    print(f"📷 使用图片: {image_path}")
    print("正在分析图片...")
    
    response = model.invoke([message])
    
    print("\n🤖 描述结果：")
    print(response.content)
    
    return response.content

# ============================================================
# 示例 2：图像问答
# ============================================================

def example_2_image_qa():
    """
    基于图片进行多轮问答
    """
    print("\n" + "=" * 60)
    print("示例 2：图像问答")
    print("=" * 60)

    image_path = check_image_exists("sample.jpg")
    if not image_path:
        print("跳过此示例")
        return None
    
    questions = [
        "图片中有什么主要物体？",
        "图片的整体色调是什么？",
        "这张图片给你什么感觉？"
    ]
    
    messages = []
    
    # 首先发送图片
    initial_message = create_image_message(
        text="我会问你关于这张图片的一些问题。",
        image_path=image_path
    )
    messages.append(initial_message)
    
    print(f"📷 已加载图片: {image_path}")
    
    for question in questions:
        print(f"\n❓ 问题: {question}")
        
        messages.append(HumanMessage(content=question))
        response = model.invoke(messages)
        messages.append(response)
        
        print(f"💬 回答: {response.content}")
    
    return messages

# ============================================================
# 示例 3：OCR 文字识别
# ============================================================

def example_3_ocr():
    """
    从图像中提取文字
    """
    print("\n" + "=" * 60)
    print("示例 3：OCR 文字识别")
    print("=" * 60)

    # 需要一张包含文字的图片
    image_path = check_image_exists("text_image.jpg")
    if not image_path:
        print("提示: 请准备一张包含文字的图片用于 OCR 测试")
        print("跳过此示例")
        return None
    
    message = create_image_message(
        text="""请仔细查看这张图片，执行以下任务：
1. 描述图片的主要内容
2. 提取图片中所有可见的文字
3. 说明这是什么类型的图片（照片、截图、文档等）

用中文回复。""",
        image_path=image_path
    )
    
    print(f"📷 使用图片: {image_path}")
    print("正在进行 OCR 识别...")
    
    response = model.invoke([message])
    
    print("\n📝 识别结果：")
    print(response.content)
    
    return response.content

# ============================================================
# 示例 4：图表分析
# ============================================================

def example_4_chart_analysis():
    """
    分析图表数据
    """
    print("\n" + "=" * 60)
    print("示例 4：图表分析")
    print("=" * 60)

    # 需要一张图表图片
    image_path = check_image_exists("chart.png")
    if not image_path:
        print("提示: 请准备一张图表图片（柱状图、折线图等）")
        print("跳过此示例")
        return None
    
    message = create_image_message(
        text="""请分析这个图表：
1. 这是什么类型的图表？
2. 图表展示了什么数据或信息？
3. 你能从图表中得出什么结论？
4. 如果有数值，请尽可能提取关键数据点

用中文详细回答。""",
        image_path=image_path
    )
    
    print(f"📷 使用图片: {image_path}")
    print("正在分析图表...")
    
    response = model.invoke([message])
    
    print("\n📊 分析结果：")
    print(response.content)
    
    return response.content

# ============================================================
# 示例 5：自定义图片分析
# ============================================================

def example_5_custom_analysis(image_path: str, prompt: str):
    """
    分析用户指定的图片
    
    Args:
        image_path: 图片路径
        prompt: 分析提示
    """
    print("\n" + "=" * 60)
    print("示例 5：自定义图片分析")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return None
    
    message = create_image_message(
        text=prompt,
        image_path=image_path
    )
    
    print(f"📷 使用图片: {image_path}")
    print(f"📝 提示: {prompt}")
    print("正在分析...")
    
    response = model.invoke([message])
    
    print("\n🤖 分析结果：")
    print(response.content)
    
    return response.content

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("图像输入教程")
    print("=" * 60)

    print("""
⚠️ 使用前请确保：
1. 已在 .env 中设置 ZHIPUAI_API_KEY
2. 已在 images/ 目录下放置测试图片:
   - sample.jpg: 任意测试图片
   - text_image.jpg: 包含文字的图片
   - chart.png: 图表图片

如果没有准备图片，示例将被跳过。
""")
    
    # 创建图片目录（如果不存在）
    IMAGES_DIR.mkdir(exist_ok=True)
    
    # 运行示例
    example_1_image_description()
    example_2_image_qa()
    example_3_ocr()
    example_4_chart_analysis()
    
    # 示例 5：自定义分析（需要用户提供图片路径）
    # 取消下面的注释来使用
    # example_5_custom_analysis(
    #     image_path="path/to/your/image.jpg",
    #     prompt="请描述这张图片"
    # )
    
    print("\n" + "=" * 60)
    print("✅ 教程运行完成！")
    print("=" * 60)
    print("""
💡 提示：
- 如需使用其他图片，请修改代码中的图片路径
- 可以调用 example_5_custom_analysis() 分析任意图片
- 确保使用支持视觉的模型（如 glm-4v）
""")
