"""
Streamlit 前端主程序

提供以下功能页面：
- 聊天界面（默认）
- 文档上传
- 知识库管理
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import streamlit as st

# 加载环境变量
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 配置页面
st.set_page_config(
    page_title="多代理智能客服系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 配置 ====================

# API 基础 URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# 会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"


# ==================== 辅助函数 ====================

def send_chat_message(message: str) -> Dict[str, Any]:
    """发送聊天消息到后端（同步）"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"message": message},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"请求失败: {str(e)}")
        return None


def send_chat_message_stream(message: str):
    """
    发送聊天消息到后端（真正的流式输出）

    返回生成器，产生流式文本内容
    不传输元数据，避免JSON被分割的问题
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat/stream",
            json={"message": message},
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')

                # 只处理以 "data: " 开头的 SSE 行
                if line_str.startswith('data: '):
                    json_str = line_str[6:]  # 移除 "data: " 前缀

                    # 跳过空的 JSON 字符串
                    if not json_str or json_str.strip() == '':
                        continue

                    try:
                        chunk = json.loads(json_str)
                        chunk_type = chunk.get("type", "unknown")

                        if chunk_type == "content":
                            # 真正的流式内容（逐 token）
                            content = chunk.get("content", "")
                            if content:  # 只 yield 非空内容
                                yield content

                        elif chunk_type == "error":
                            yield f"错误: {chunk.get('error', '未知错误')}"
                            return  # 使用 return 确保完全退出

                        # 忽略 intent 和 final 类型，我们通过同步接口获取元数据

                    except json.JSONDecodeError as e:
                        # 跳过无法解析的行，继续处理
                        print(f"[DEBUG] JSON 解析失败: {e}, 原始数据: {repr(json_str)}")
                        continue
                    except Exception as e:
                        print(f"[DEBUG] 处理异常: {e}")
                        yield f"处理错误: {str(e)}"
                        return

    except requests.exceptions.RequestException as e:
        yield f"网络错误: {str(e)}"
    except Exception as e:
        yield f"未知错误: {str(e)}"


def upload_document(file, category: str) -> Dict[str, Any]:
    """上传文档"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"category": category}

        response = requests.post(
            f"{API_BASE_URL}/api/upload",
            files=files,
            data=data,
            timeout=300  # 增加到 5 分钟
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"上传失败: {str(e)}")
        return None


def get_knowledge_list(category: str = None) -> List[Dict[str, Any]]:
    """获取知识库列表"""
    try:
        params = {"category": category} if category else {}
        response = requests.get(
            f"{API_BASE_URL}/api/knowledge",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data["documents"]
    except Exception as e:
        st.error(f"❌ 获取知识库列表失败: {str(e)}")
        return []


def get_statistics() -> Dict[str, Any]:
    """获取统计信息"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/stats",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ 获取统计信息失败: {str(e)}")
        return {}


def delete_document(doc_id: str) -> bool:
    """删除文档"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/knowledge/{doc_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()["status"] == "success"
    except Exception as e:
        st.error(f"❌ 删除失败: {str(e)}")
        return False


# ==================== 页面：聊天界面 ====================

def chat_page():
    """聊天界面"""
    st.title("多代理智能客服系统")
    st.markdown("---")

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 如果是最新消息，显示额外信息
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_msg = st.session_state.messages[-1]
        if "metadata" in last_msg:
            metadata = last_msg["metadata"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("意图", metadata.get("intent", "N/A"))
            with col2:
                st.metric("置信度", f"{metadata.get('confidence', 0):.2f}")
            with col3:
                st.metric("质量评分", f"{metadata.get('quality_score', 0):.2f}")

            if metadata.get("escalated"):
                st.warning("此对话已升级到人工客服")

    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 添加到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 创建助手消息占位符
        with st.chat_message("assistant"):
            # 使用 st.empty() 创建可更新的占位符
            response_placeholder = st.empty()

            # 收集完整响应
            full_response = ""

            # 流式显示响应
            try:
                for chunk in send_chat_message_stream(prompt):
                    full_response += chunk
                    # 实时更新显示
                    response_placeholder.markdown(full_response + "▌")

                # 移除光标
                response_placeholder.markdown(full_response)

                # 添加到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                import traceback
                print(f"[DEBUG] 前端异常: {e}")
                traceback.print_exc()
                response_placeholder.error(f"请求失败: {str(e)}")

    # 清空历史按钮
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()


# ==================== 页面：文档上传 ====================

def upload_page():
    """文档上传页面"""
    st.title("📚 知识库管理")
    st.markdown("---")

    # 上传表单
    st.subheader("📤 上传新文档")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "选择文档",
            type=["pdf", "txt", "md"],
            help="支持 PDF、TXT、Markdown 格式"
        )

        category = st.radio(
            "选择知识库类别",
            options=["products", "technical", "faq"],
            format_func=lambda x: {
                "products": "🛍️ 产品文档",
                "technical": "🔧 技术文档",
                "faq": "❓ FAQ"
            }[x],
            horizontal=True
        )

        if uploaded_file and st.button("🚀 上传文档", type="primary"):
            with st.spinner("正在处理文档..."):
                result = upload_document(uploaded_file, category)

            if result and result.get("status") == "success":
                st.success(f"""
                ✅ 上传成功！
                - 文档ID: {result['document_id']}
                - 创建块数: {result['chunks']}
                - 类别: {result['category']}
                """)
            elif result:
                st.error(f"❌ {result.get('message', '上传失败')}")

    with col2:
        st.info("""
        💡 上传说明：

        1. **产品文档**: 产品手册、价格表、功能介绍
        2. **技术文档**: 故障排除、维修指南、技术参数
        3. **FAQ**: 常见问题、注意事项、使用提示

        📝 支持格式：
        - PDF 文档
        - TXT 文本
        - Markdown 文档

        ⚠️ 注意：
        - 文件大小限制: 10MB
        - 建议上传清晰、结构化的文档
        """)

    st.markdown("---")

    # 知识库列表
    st.subheader("📋 当前知识库")

    # 统计信息
    stats = get_statistics()
    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总文档数", stats["total_documents"])
        with col2:
            st.metric("知识库类别", len(stats.get("categories", [])))
        with col3:
            st.metric("最新类别", stats.get("categories", ["N/A"])[-1] if stats.get("categories") else "N/A")

    # 文档列表
    category_filter = st.selectbox(
        "筛选类别",
        options=["全部", "products", "technical", "faq"]
    )

    docs = get_knowledge_list(
        category=None if category_filter == "全部" else category_filter
    )

    if docs:
        st.write(f"共 {len(docs)} 个文档")

        for doc in docs:
            with st.expander(f"📄 {doc['filename']} ({doc['category']})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**ID**: {doc['id']}")
                    st.write(f"**大小**: {doc['size'] / 1024:.1f} KB")
                    st.write(f"**上传时间**: {doc['upload_date'][:10]}")
                with col2:
                    if st.button("🗑️ 删除", key=f"delete_{doc['id']}", use_container_width=True):
                        if delete_document(doc['id']):
                            st.success("已删除")
                            st.rerun()
    else:
        st.info("暂无文档")


# ==================== 侧边栏 ====================

def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.title("🤖 智能客服系统")
        st.markdown("---")

        # 页面导航
        page = st.radio(
            "选择页面",
            ["💬 聊天", "📚 知识库管理"],
            format_func=lambda x: x.split(" ", 1)[1],
            key="page_radio"
        )

        st.session_state.current_page = "chat" if page == "💬 聊天" else "knowledge"

        st.markdown("---")

        # 系统状态
        st.subheader("📊 系统状态")

        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ 后端服务正常")
            else:
                st.error("❌ 后端服务异常")
        except:
            st.error("❌ 无法连接后端服务")

        st.markdown("---")

        # 使用说明
        st.subheader("📖 使用说明")
        st.info("""
        **功能说明**：

        💬 **聊天页面**
        - 输入问题进行智能问答
        - 系统自动识别意图并路由
        - 支持技术支持、订单、产品咨询

        📚 **知识库管理**
        - 上传文档扩充知识库
        - 管理已有文档
        - 查看统计信息

        **支持的意图**：
        - 🔧 技术问题
        - 📦 订单查询
        - 🛍️ 产品咨询
        """)


# ==================== 主程序 ====================

def main():
    """主程序"""
    # 侧边栏
    sidebar()

    # 根据选择显示不同页面
    if st.session_state.current_page == "chat":
        chat_page()
    else:
        upload_page()


if __name__ == "__main__":
    main()
