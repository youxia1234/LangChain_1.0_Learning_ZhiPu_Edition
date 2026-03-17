#!/bin/bash
# 多代理智能客服系统启动脚本 (Unix/Linux/macOS)

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}多代理智能客服系统${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3${NC}"
    echo "请先安装 Python 3.10+"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python 版本: $(python3 --version)"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}警告: 未找到 .env 文件${NC}"
    echo "正在从 .env.example 复制..."
    cp .env.example .env
    echo ""
    echo -e "${YELLOW}重要: 请编辑 .env 文件并填入你的 API Keys${NC}"
    echo "       - ZHIPUAI_API_KEY"
    echo "       - PINECONE_API_KEY"
    echo ""
    read -p "按 Enter 继续..."
fi

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 依赖安装失败${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} 依赖检查完成"
echo ""
echo -e "${BLUE}========================================${NC}"
echo "启动服务..."
echo -e "${BLUE}========================================${NC}"
echo ""

# 启动后端
echo -e "[1/2] ${GREEN}启动后端服务 (FastAPI)...${NC}"
cd backend
python3 main.py &
BACKEND_PID=$!
cd ..

# 等待后端启动
echo "等待后端服务启动..."
sleep 3

# 检查后端是否启动成功
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓${NC} 后端服务已启动 (PID: $BACKEND_PID)"
else
    echo -e "${RED}错误: 后端服务启动失败${NC}"
    exit 1
fi

# 启动前端
echo -e "[2/2] ${GREEN}启动前端界面 (Streamlit)...${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}系统启动完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "前端地址: ${GREEN}http://localhost:8501${NC}"
echo -e "后端地址: ${GREEN}http://localhost:8000${NC}"
echo -e "API 文档: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
echo ""

# 设置陷阱，确保退出时关闭后端
trap "echo ''; echo '正在停止服务...'; kill $BACKEND_PID 2>/dev/null; exit 0" INT TERM

# 启动前端
cd frontend
streamlit run main.py

# 清理
kill $BACKEND_PID 2>/dev/null
