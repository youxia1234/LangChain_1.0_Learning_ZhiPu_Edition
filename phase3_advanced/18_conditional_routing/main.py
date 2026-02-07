"""
模块 18：条件路由
学习如何使用条件边实现动态工作流控制
"""

import os
import sys
import random
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI）
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ============================================================
# 示例 1：评分路由系统
# ============================================================

def score_based_routing():
    """
    根据评分决定处理流程
    - 优秀 (>= 90)：发送表扬信
    - 良好 (>= 70)：正常通过
    - 需改进 (>= 50)：提供建议
    - 不合格 (< 50)：需要重新提交
    """
    print("\n" + "=" * 60)
    print("示例 1：评分路由系统")
    print("=" * 60)

    class ScoreState(TypedDict):
        content: str
        score: int
        feedback: str
        result: str

    
    def evaluate(state: ScoreState) -> dict:
        """评估内容并打分"""
        messages = [
            SystemMessage(content="""你是一个内容评估专家。请评估以下内容并给出1-100的分数。
只返回一个数字分数，不要其他内容。评估标准：
- 90-100：优秀，内容完整、准确、有创意
- 70-89：良好，内容基本完整，表达清晰
- 50-69：需改进，内容有所欠缺
- 0-49：不合格，需要重新撰写"""),
            HumanMessage(content=state["content"])
        ]
        response = model.invoke(messages)
        
        try:
            score = int(response.content.strip())
            score = max(0, min(100, score))  # 确保在 0-100 范围内
        except:
            score = 70  # 默认分数
        
        print(f"  [评估] 评分：{score}")
        return {"score": score}

    def route_by_score(state: ScoreState) -> Literal["excellent", "good", "improve", "reject"]:
        """根据分数路由"""
        score = state["score"]
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "improve"
        else:
            return "reject"

    def handle_excellent(state: ScoreState) -> dict:
        """处理优秀评分"""
        print("  [优秀] 🌟 发送表扬通知")
        return {
            "feedback": "恭喜！您的内容非常出色！",
            "result": "APPROVED_WITH_HONORS"
        }

    def handle_good(state: ScoreState) -> dict:
        """处理良好评分"""
        print("  [良好] ✅ 正常通过")
        return {
            "feedback": "内容合格，已通过审核。",
            "result": "APPROVED"
        }

    def handle_improve(state: ScoreState) -> dict:
        """处理需改进评分"""
        print("  [需改进] 📝 生成改进建议")
        messages = [
            SystemMessage(content="请为以下内容提供简洁的改进建议（50字以内）。用中文回复。"),
            HumanMessage(content=state["content"])
        ]
        response = model.invoke(messages)
        return {
            "feedback": f"建议改进：{response.content}",
            "result": "NEEDS_IMPROVEMENT"
        }

    def handle_reject(state: ScoreState) -> dict:
        """处理不合格评分"""
        print("  [不合格] ❌ 需要重新提交")
        return {
            "feedback": "内容不符合要求，请重新撰写并提交。",
            "result": "REJECTED"
        }

    # 构建图
    graph = StateGraph(ScoreState)
    
    graph.add_node("evaluate", evaluate)
    graph.add_node("excellent", handle_excellent)
    graph.add_node("good", handle_good)
    graph.add_node("improve", handle_improve)
    graph.add_node("reject", handle_reject)
    
    graph.add_edge(START, "evaluate")
    
    graph.add_conditional_edges(
        "evaluate",
        route_by_score,
        {
            "excellent": "excellent",
            "good": "good",
            "improve": "improve",
            "reject": "reject"
        }
    )
    
    for node in ["excellent", "good", "improve", "reject"]:
        graph.add_edge(node, END)
    
    app = graph.compile()
    
    # 测试不同质量的内容
    test_contents = [
        "Python 是一种广泛使用的高级编程语言，以其清晰的语法和强大的功能著称。它支持多种编程范式，包括面向对象、函数式和过程式编程。Python 拥有丰富的标准库和第三方库，广泛应用于 Web 开发、数据科学、人工智能等领域。",
        "Python 是编程语言，很好用。",
        "编程"
    ]
    
    for content in test_contents:
        print(f"\n提交内容: {content[:30]}...")
        result = app.invoke({"content": content})
        print(f"结果: {result['result']}")
        print(f"反馈: {result['feedback']}")

# ============================================================
# 示例 2：重试机制
# ============================================================

def retry_mechanism():
    """
    实现带重试的工作流
    - 任务可能随机失败
    - 最多重试 3 次
    - 超过重试次数后使用备用方案
    """
    print("\n" + "=" * 60)
    print("示例 2：重试机制")
    print("=" * 60)

    class RetryState(TypedDict):
        task: str
        retry_count: int
        max_retries: int
        success: bool
        result: str
        error_message: str

    def execute_task(state: RetryState) -> dict:
        """执行可能失败的任务"""
        retry_count = state.get("retry_count", 0)
        
        # 模拟任务执行（有50%概率失败）
        success = random.random() > 0.5
        
        if success:
            print(f"  [执行] ✅ 任务成功 (尝试 {retry_count + 1})")
            return {
                "success": True,
                "result": f"任务 '{state['task']}' 执行成功！",
                "retry_count": retry_count + 1
            }
        else:
            print(f"  [执行] ❌ 任务失败 (尝试 {retry_count + 1})")
            return {
                "success": False,
                "error_message": "模拟的随机错误",
                "retry_count": retry_count + 1
            }

    def should_retry(state: RetryState) -> Literal["retry", "fallback", "success"]:
        """决定是否重试"""
        if state["success"]:
            return "success"
        
        if state["retry_count"] < state["max_retries"]:
            print(f"  [路由] 准备第 {state['retry_count'] + 1} 次重试...")
            return "retry"
        
        print("  [路由] 重试次数已达上限，使用备用方案")
        return "fallback"

    def success_handler(state: RetryState) -> dict:
        """成功处理"""
        return {"result": f"✅ 最终结果：{state['result']}"}

    def fallback_handler(state: RetryState) -> dict:
        """备用方案"""
        print("  [备用] 执行备用方案...")
        return {
            "result": f"⚠️ 使用备用方案完成任务（原任务失败 {state['retry_count']} 次）"
        }

    # 构建图
    graph = StateGraph(RetryState)
    
    graph.add_node("execute", execute_task)
    graph.add_node("success", success_handler)
    graph.add_node("fallback", fallback_handler)
    
    graph.add_edge(START, "execute")
    
    graph.add_conditional_edges(
        "execute",
        should_retry,
        {
            "retry": "execute",      # 重试：回到执行节点
            "fallback": "fallback",  # 备用方案
            "success": "success"     # 成功
        }
    )
    
    graph.add_edge("success", END)
    graph.add_edge("fallback", END)
    
    app = graph.compile()
    
    # 运行多次测试
    for i in range(3):
        print(f"\n--- 测试 {i + 1} ---")
        result = app.invoke({
            "task": "发送通知邮件",
            "retry_count": 0,
            "max_retries": 3,
            "success": False
        })
        print(f"结果: {result['result']}")

# ============================================================
# 示例 3：复杂决策树
# ============================================================

def complex_decision_tree():
    """
    复杂决策树：多条件组合
    模拟贷款审批流程
    """
    print("\n" + "=" * 60)
    print("示例 3：复杂决策树 - 贷款审批")
    print("=" * 60)

    class LoanState(TypedDict):
        applicant_name: str
        credit_score: int
        income: int
        loan_amount: int
        has_collateral: bool
        current_stage: str
        decision: str
        reason: str

    def initial_check(state: LoanState) -> dict:
        """初步检查"""
        print(f"  [初步检查] 申请人: {state['applicant_name']}")
        print(f"    - 信用分: {state['credit_score']}")
        print(f"    - 月收入: ¥{state['income']}")
        print(f"    - 贷款金额: ¥{state['loan_amount']}")
        print(f"    - 有抵押物: {state['has_collateral']}")
        return {"current_stage": "initial_check_done"}

    def route_initial(state: LoanState) -> Literal["auto_reject", "credit_review", "income_review"]:
        """初步路由"""
        # 信用分太低直接拒绝
        if state["credit_score"] < 550:
            return "auto_reject"
        
        # 高信用分走快速通道
        if state["credit_score"] >= 750:
            return "income_review"
        
        # 中等信用分需要详细审查
        return "credit_review"

    def auto_reject(state: LoanState) -> dict:
        """自动拒绝"""
        print("  [自动拒绝] 信用分过低")
        return {
            "decision": "REJECTED",
            "reason": "信用评分低于最低要求"
        }

    def credit_review(state: LoanState) -> dict:
        """信用审查"""
        print("  [信用审查] 进行详细信用评估...")
        return {"current_stage": "credit_reviewed"}

    def route_credit(state: LoanState) -> Literal["income_review", "manual_review"]:
        """信用审查后路由"""
        # 有抵押物可以继续
        if state["has_collateral"]:
            return "income_review"
        # 无抵押物需要人工审核
        return "manual_review"

    def income_review(state: LoanState) -> dict:
        """收入审查"""
        print("  [收入审查] 评估还款能力...")
        return {"current_stage": "income_reviewed"}

    def route_income(state: LoanState) -> Literal["approve", "partial_approve", "manual_review"]:
        """收入审查后路由"""
        # 计算贷款收入比
        loan_to_income = state["loan_amount"] / (state["income"] * 12)
        
        if loan_to_income <= 3:  # 贷款金额不超过年收入的3倍
            return "approve"
        elif loan_to_income <= 5:
            return "partial_approve"
        else:
            return "manual_review"

    def approve(state: LoanState) -> dict:
        """批准"""
        print("  [批准] ✅ 贷款申请通过！")
        return {
            "decision": "APPROVED",
            "reason": "符合所有审批条件"
        }

    def partial_approve(state: LoanState) -> dict:
        """部分批准"""
        approved_amount = state["income"] * 12 * 3  # 批准年收入3倍
        print(f"  [部分批准] ⚠️ 批准部分金额: ¥{approved_amount}")
        return {
            "decision": "PARTIALLY_APPROVED",
            "reason": f"批准金额：¥{approved_amount}（原申请：¥{state['loan_amount']}）"
        }

    def manual_review(state: LoanState) -> dict:
        """人工审核"""
        print("  [人工审核] 📋 已转人工审核")
        return {
            "decision": "PENDING_REVIEW",
            "reason": "需要信贷专员进一步审核"
        }

    # 构建图
    graph = StateGraph(LoanState)
    
    graph.add_node("initial_check", initial_check)
    graph.add_node("auto_reject", auto_reject)
    graph.add_node("credit_review", credit_review)
    graph.add_node("income_review", income_review)
    graph.add_node("approve", approve)
    graph.add_node("partial_approve", partial_approve)
    graph.add_node("manual_review", manual_review)
    
    graph.add_edge(START, "initial_check")
    
    graph.add_conditional_edges("initial_check", route_initial, {
        "auto_reject": "auto_reject",
        "credit_review": "credit_review",
        "income_review": "income_review"
    })
    
    graph.add_conditional_edges("credit_review", route_credit, {
        "income_review": "income_review",
        "manual_review": "manual_review"
    })
    
    graph.add_conditional_edges("income_review", route_income, {
        "approve": "approve",
        "partial_approve": "partial_approve",
        "manual_review": "manual_review"
    })
    
    for node in ["auto_reject", "approve", "partial_approve", "manual_review"]:
        graph.add_edge(node, END)
    
    app = graph.compile()
    
    # 测试不同的申请案例
    test_cases = [
        {"applicant_name": "张三", "credit_score": 800, "income": 20000, "loan_amount": 500000, "has_collateral": True},
        {"applicant_name": "李四", "credit_score": 650, "income": 10000, "loan_amount": 200000, "has_collateral": True},
        {"applicant_name": "王五", "credit_score": 500, "income": 8000, "loan_amount": 100000, "has_collateral": False},
        {"applicant_name": "赵六", "credit_score": 720, "income": 15000, "loan_amount": 1000000, "has_collateral": False},
    ]
    
    for case in test_cases:
        print(f"\n{'='*40}")
        result = app.invoke(case)
        print(f"\n决定: {result['decision']}")
        print(f"原因: {result['reason']}")

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("条件路由教程")
    print("=" * 60)
    
    score_based_routing()
    retry_mechanism()
    complex_decision_tree()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
