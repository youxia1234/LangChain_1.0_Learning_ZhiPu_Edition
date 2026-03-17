@echo off
REM 多代理智能客服系统启动脚本 (Windows)

echo ========================================
echo 多代理智能客服系统
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python
    echo 请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 检查 .env 文件
if not exist ".env" (
    echo 警告: 未找到 .env 文件
    echo 正在从 .env.example 复制...
    copy .env.example .env >nul
    echo.
    echo 重要: 请编辑 .env 文件并填入你的 API Keys
    echo        - ZHIPUAI_API_KEY
    echo        - PINECONE_API_KEY
    echo.
    pause
)

REM 检查依赖
echo 检查依赖...
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装依赖...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo 启动服务...
echo ========================================
echo.

REM 启动后端 (新窗口)
echo [1/2] 启动后端服务 (FastAPI)...
start "Multi-Agent Backend" cmd /k "cd backend && python main.py"

REM 等待后端启动
echo 等待后端服务启动...
timeout /t 3 /nobreak >nul

REM 启动前端
echo [2/2] 启动前端界面 (Streamlit)...
echo.
echo ========================================
echo 系统启动完成！
echo ========================================
echo.
echo 前端地址: http://localhost:8501
echo 后端地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo.

cd frontend
streamlit run main.py

pause
