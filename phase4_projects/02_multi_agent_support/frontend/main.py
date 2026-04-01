"""
企业智能问答系统 - 前端主程序

面向制造企业官网访客的智能问答界面：
- 智能对话（支持流式输出）
- 知识库管理
- 系统状态监控
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import streamlit as st

from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 配置页面
st.set_page_config(
    page_title="华智精密 · 智能问答助手",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 加载自定义样式 ====================

def load_custom_styles():
    """加载自定义 CSS 样式"""
    css_path = Path(__file__).parent / "static" / "styles.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # 使用 Streamlit 默认样式

load_custom_styles()

# ==================== 配置 ====================

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
    """发送聊天消息到后端（流式输出）"""
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

                if line_str.startswith('data: '):
                    json_str = line_str[6:]

                    if not json_str or json_str.strip() == '':
                        continue

                    try:
                        chunk = json.loads(json_str)
                        chunk_type = chunk.get("type", "unknown")

                        if chunk_type == "content":
                            content = chunk.get("content", "")
                            if content:
                                yield content

                        elif chunk_type == "error":
                            yield f"错误: {chunk.get('error', '未知错误')}"
                            return

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
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
            timeout=300
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
        st.error(f"获取知识库列表失败: {str(e)}")
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
        st.error(f"删除失败: {str(e)}")
        return False


# ==================== 页面：聊天界面 ====================

def chat_page():
    """聊天界面"""
    # 页面标题
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; color: #0f172a;">智能问答助手</h1>
            <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.875rem;">随时为您解答关于产品、技术、合作的问题</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 欢迎引导
    if not st.session_state.messages:
        st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 16px; border: 1px solid #bae6fd; margin-bottom: 1.5rem;">
            <h3 style="margin: 0 0 0.75rem 0; color: #0369a1;">欢迎来到华智精密制造！</h3>
            <p style="margin: 0 0 1rem 0; color: #0c4a6e; font-size: 0.9rem;">
                我是您的智能问答助手，可以帮您了解我们的产品、技术实力和合作方式。试试以下问题：
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 推荐问题按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📡 你们有哪些产品？", use_container_width=True):
                st.session_state.quick_query = "你们有哪些产品？给我介绍一下"
            if st.button("🏆 公司有什么资质认证？", use_container_width=True):
                st.session_state.quick_query = "公司有什么资质认证？通过了几项体系认证？"
        with col2:
            if st.button("🔧 技术实力怎么样？", use_container_width=True):
                st.session_state.quick_query = "你们的技术实力怎么样？研发能力如何？"
            if st.button("🤝 支持OEM代工吗？", use_container_width=True):
                st.session_state.quick_query = "你们支持OEM代工吗？最低起订量是多少？"

        st.markdown("---")

    # 聊天统计
    if st.session_state.messages:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("对话轮次", len([m for m in st.session_state.messages if m["role"] == "user"]))
        with col2:
            st.metric("消息总数", len(st.session_state.messages))
        with col3:
            st.metric("会话时长", "进行中")

        st.markdown("---")

    # 聊天容器
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_msg = st.session_state.messages[-1]
            if "metadata" in last_msg:
                metadata = last_msg["metadata"]
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("意图识别", metadata.get("intent", "N/A"))
                with col2:
                    st.metric("置信度", f"{metadata.get('confidence', 0):.0%}")
                with col3:
                    st.metric("质量评分", f"{metadata.get('quality_score', 0):.1f}/10")

                if metadata.get("escalated"):
                    st.warning("此对话已建议联系人工客服")

    # 处理快捷问题
    query_to_send = getattr(st.session_state, 'quick_query', None)
    if query_to_send:
        del st.session_state.quick_query
        prompt = query_to_send
    else:
        prompt = st.chat_input("请输入您的问题...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in send_chat_message_stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                import traceback
                print(f"[DEBUG] 前端异常: {e}")
                traceback.print_exc()
                response_placeholder.error(f"请求失败: {str(e)}")

    # 操作栏
    if st.session_state.messages:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("清空对话历史", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("重新开始", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


# ==================== 页面：知识库管理 ====================

def knowledge_page():
    """知识库管理页面"""
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; color: #0f172a;">知识库管理</h1>
            <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.875rem;">上传文档 · 管理知识库 · 查看统计</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 统计概览
    stats = get_statistics()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总文档", stats.get("total_documents", 0))
        with col2:
            st.metric("类别数", len(stats.get("categories", [])))
        with col3:
            st.metric("产品文档", stats.get("category_stats", {}).get("products", 0))
        with col4:
            st.metric("技术文档", stats.get("category_stats", {}).get("capabilities", 0))

    st.markdown("---")

    # 上传表单
    st.markdown('<p style="color: #0f172a; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">上传新文档</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "选择文档",
            type=["pdf", "txt", "md"],
            help="支持 PDF、TXT、Markdown 格式",
            label_visibility="visible"
        )

        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.markdown(f"""
            <div style="padding: 0.75rem; background: #f1f5f9; border-radius: 8px; border: 1px solid #e2e8f0;">
                <p style="margin: 0; color: #475569; font-size: 0.875rem;">
                    <strong>文件名:</strong> {uploaded_file.name}<br>
                    <strong>大小:</strong> {file_size:.1f} KB
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p style="color: #64748b; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem;">选择知识库类别</p>', unsafe_allow_html=True)

        category = st.radio(
            "",
            options=["products", "capabilities", "company", "faq"],
            format_func=lambda x: {
                "products": "产品文档（规格书/选型指南）",
                "capabilities": "技术实力（认证/工艺）",
                "company": "公司介绍（历程/荣誉）",
                "faq": "常见问题"
            }[x],
            horizontal=True,
            label_visibility="collapsed"
        )

        if uploaded_file and st.button("上传文档", type="primary", use_container_width=True):
            with st.spinner("正在处理文档，请稍候..."):
                result = upload_document(uploaded_file, category)

            if result and result.get("status") == "success":
                st.success(f"上传成功！文档ID: {result['document_id']}，创建 {result['chunks']} 个块")
            elif result:
                st.error(f"上传失败: {result.get('message', '未知错误')}")

    with col2:
        st.markdown("""
        <div style="padding: 1.25rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px; border: 1px solid #bae6fd;">
            <p style="margin: 0 0 0.75rem 0; color: #0369a1; font-weight: 600; font-size: 0.875rem;">上传说明</p>
            <p style="margin: 0; color: #0c4a6e; font-size: 0.8rem; line-height: 1.6;">
                <strong>产品文档</strong>: 产品手册、规格书、选型指南<br><br>
                <strong>技术实力</strong>: 认证证书、工艺介绍、检测报告<br><br>
                <strong>公司介绍</strong>: 公司简介、发展历程、荣誉资质<br><br>
                <strong>FAQ</strong>: 常见问题、注意事项
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 知识库列表
    st.markdown('<p style="color: #0f172a; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">当前知识库</p>', unsafe_allow_html=True)

    category_filter = st.selectbox(
        "筛选类别",
        options=["全部", "products", "capabilities", "company", "faq"],
        label_visibility="collapsed"
    )

    docs = get_knowledge_list(
        category=None if category_filter == "全部" else category_filter
    )

    if docs:
        st.markdown(f'<p style="color: #64748b; font-size: 0.875rem;">共 {len(docs)} 个文档</p>', unsafe_allow_html=True)

        for doc in docs:
            category_emoji = {
                "products": "📦",
                "capabilities": "🔧",
                "company": "🏢",
                "faq": "❓"
            }.get(doc["category"], "📄")

            with st.expander(f"{category_emoji} {doc['filename']} · {doc['category']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**ID:** {doc['id']}  |  **大小:** {doc.get('size', 0) / 1024:.1f} KB  |  **上传时间:** {doc.get('upload_date', 'N/A')[:10]}")
                with col2:
                    if st.button("删除", key=f"delete_{doc['id']}", use_container_width=True):
                        if delete_document(doc['id']):
                            st.success("文档已删除")
                            st.rerun()
    else:
        st.info("暂无文档，请上传文档到知识库")


# ==================== 侧边栏 ====================

def sidebar():
    """侧边栏"""
    with st.sidebar:
        # 品牌标识
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin: 0; font-size: 1.6rem; color: #3b82f6;">🏭 华智精密制造</h1>
            <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.8rem;">Enterprise Intelligent QA System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 页面导航
        st.markdown('<p style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;">导航菜单</p>', unsafe_allow_html=True)

        page = st.radio(
            "",
            ["💬 智能问答", "📚 知识库管理"],
            format_func=lambda x: x.split(" ", 1)[1],
            key="page_radio",
            label_visibility="collapsed"
        )

        st.session_state.current_page = "chat" if page == "💬 智能问答" else "knowledge"

        st.markdown("---")

        # 系统状态
        st.markdown('<p style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;">系统状态</p>', unsafe_allow_html=True)

        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px;">
                    <span style="color: #10b981;">●</span>
                    <span style="color: #f8fafc;">后端服务正常</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
                    <span style="color: #ef4444;">●</span>
                    <span style="color: #f8fafc;">后端服务异常</span>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
                <span style="color: #ef4444;">●</span>
                <span style="color: #f8fafc;">无法连接后端</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 快捷统计
        try:
            stats = get_statistics()
            if stats:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("文档", stats.get("total_documents", 0))
                with col2:
                    st.metric("类别", len(stats.get("categories", [])))
        except:
            pass

        st.markdown("---")

        # 关于
        with st.expander("关于系统"):
            st.markdown("""
            **华智精密 · 智能问答助手**

            基于 LangChain 1.0 + LangGraph 构建的企业智能问答系统。

            **功能亮点：**
            - 多代理协同问答
            - 混合检索 (BM25 + 向量)
            - 查询重写与重排序
            - 引用溯源

            **RAG 五层优化：**
            1. 查询重写 (Query Rewriting)
            2. 语义分块 + 元数据增强
            3. LLM 重排序 + 过滤
            4. 引用溯源 + 幻觉检查
            5. BM25 持久化 + 搜索缓存
            """)

        # 底部信息
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #64748b; font-size: 0.75rem;">
            <p>© 2026 华智精密制造有限公司</p>
        </div>
        """, unsafe_allow_html=True)


# ==================== 主程序 ====================

def main():
    """主程序"""
    sidebar()

    if st.session_state.current_page == "chat":
        chat_page()
    else:
        knowledge_page()


if __name__ == "__main__":
    main()
