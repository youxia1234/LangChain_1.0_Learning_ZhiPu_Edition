# 发现与决策

## 需求
1. **性能优化**：提升系统响应速度，减少 LLM 调用延迟
2. **UI 美化**：将 Streamlit 默认界面升级为现代化、专业化的产品外观
3. **功能完善**：添加生产级系统必备的功能（会话持久化、监控等）
4. **部署友好**：简化部署流程，提供 Docker 支持

## 研究发现

### 当前系统架构分析

**后端 (FastAPI)**：
- ✅ 良好的模块化设计（agents、rag、knowledge 分离）
- ✅ 支持混合检索（BM25 + Vector）
- ⚠️ 同步 API 处理（无异步优化）
- ⚠️ 无缓存机制（每次查询都调用 LLM）
- ⚠️ 无会话持久化（刷新页面丢失历史）

**前端 (Streamlit)**：
- ✅ 基本功能完整（聊天、上传、管理）
- ⚠️ 使用默认 Streamlit 样式（看起来像 demo）
- ⚠️ 缺少加载动画和过渡效果
- ⚠️ 侧边栏布局不够专业
- ⚠️ 无暗色模式

### 性能瓶颈识别

| 组件 | 瓶颈 | 影响 | 优先级 |
|------|------|------|--------|
| LLM 调用 | 无缓存，重复查询 | 高 | P0 |
| RAG 检索 | 同步处理，无批量 | 中 | P1 |
| 文档上传 | 单线程处理 | 中 | P1 |
| 数据库连接 | 无连接池 | 低 | P2 |

### UI 改进方向

**当前问题**：
- 使用 Streamlit 默认组件样式
- 缺乏品牌标识和视觉层次
- 聊天界面简陋
- 无加载状态反馈

**改进方案**：
- 自定义 CSS 样式系统
- 现代化配色方案
- 专业聊天界面设计
- 动画和过渡效果
- 响应式布局

## 技术决策
| 决策 | 理由 |
|------|------|
| 保留 Streamlit | 快速迭代、Python 原生、团队熟悉 |
| 自定义 CSS 覆盖 | 无需重构、成本最低 |
| 响应缓存 (ResponseCache) | 减少 LLM 调用，提升响应速度 |
| SQLite 持久化 | 轻量、无需额外服务 |
| 滑动窗口限流 | 防止 API 滥用，保护服务 |
| 缓存键 SHA256 哈希 | 避免键冲突，支持中文输入 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| Streamlit CSS 注入限制 | 使用 st.markdown() 注入 style 标签 |
| 缓存失效问题 | 使用 TTL 缓存避免过期数据 |
| 会话状态持久化 | 使用 SQLite + session_id |

## 性能优化实现

### 响应缓存系统
- **ResponseCache 类**：TTL 缓存，支持 LRU 淘汰
- **缓存键生成**：SHA256(agent_type + message)
- **缓存统计**：命中率、缓存大小、请求数
- **默认配置**：1000 条缓存，1 小时过期

### 会话持久化
- **SessionManager 类**：SQLite 存储
- **数据库路径**：backend/data/sessions.db
- **支持操作**：创建、获取、更新、列表、删除
- **自动清理**：cleanup_old_sessions() 方法

### 请求限流
- **RateLimiter 类**：滑动窗口算法
- **默认配置**：60 请求 / 分钟
- **客户端标识**：支持 IP 或 session_id
- **剩余查询**：get_remaining() 方法

### 新增 API 端点
- `GET /api/performance/cache` - 获取缓存统计
- `POST /api/performance/cache/clear` - 清空缓存
- `GET /api/performance/sessions` - 列出会话
- `GET /api/performance/sessions/{id}` - 获取会话详情
- `DELETE /api/performance/sessions/{id}` - 删除会话

## 资源
- **项目路径**: `phase4_projects/02_multi_agent_support/`
- **核心文件**:
  - `backend/main.py` - FastAPI 入口
  - `backend/core/agents.py` - 多代理系统
  - `backend/core/hybrid_rag.py` - 混合检索引擎
  - `frontend/main.py` - Streamlit 前端
- **环境变量**: `.env` (ZHIPUAI_API_KEY, 等)

## 视觉/浏览器发现
<!-- 关键：每执行2次查看/浏览器操作后必须更新此部分 -->
<!-- 多模态内容必须立即以文本形式记录 -->

### 当前 UI 截图描述
**聊天页面**：
- 标准 Streamlit 聊天组件
- 侧边栏导航
- 基础的输入框和发送按钮
- 简单的消息展示

**知识库管理页面**：
- 文件上传器
- 单选按钮选择类别
- 基础的文档列表展示
- 简单的统计指标

### 目标 UI 参考风格
- 类似 ChatGPT 的聊天界面
- 现代化卡片布局
- 柔和的阴影和圆角
- 专业的配色方案
- 流畅的动画效果

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
*防止视觉信息丢失*
