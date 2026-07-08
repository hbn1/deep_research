import ast
import json
import logging
import math
import re
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, _last_content)

logger = logging.getLogger('mult_agents')


_BINARY_OPS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
}


def _normalize_arithmetic_expression(query: str) -> str | None:
    text = query.strip()
    if not text or len(text) > 80:
        return None
    text = text.replace("×", "*").replace("÷", "/").replace("＋", "+").replace("－", "-")
    text = re.sub(r"^(answer only|calculate)\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(what is|what's)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(please\s+)?compute\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(请计算|计算|算一下|帮我算一下)\s*[:：]?\s*", "", text)
    text = text.replace("等于多少", "").replace("是多少", "")
    text = text.strip().rstrip("=?？。")
    if not text or not re.fullmatch(r"[0-9\s+\-*/%().]+", text):
        return None
    if not any(op in text for op in "+-*/%"):
        return None
    return text


def _eval_arithmetic_expression(expr: str) -> float | int:
    tree = ast.parse(expr, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            value = node.value
            if abs(float(value)) > 1e12:
                raise ValueError("number is too large")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            for op_type, operation in _BINARY_OPS.items():
                if isinstance(node.op, op_type):
                    result = operation(left, right)
                    if not math.isfinite(float(result)) or abs(float(result)) > 1e12:
                        raise ValueError("result is too large")
                    return result
        raise ValueError("unsupported expression")

    return visit(tree)


def _format_arithmetic_result(value: float | int) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _deterministic_direct_answer(query: str) -> str | None:
    stripped = query.strip()
    simple = stripped.strip(" !！.。?？").lower()
    if simple in {"hi", "hello", "hey"} or stripped in {"你好", "您好"}:
        return "你好，我是 DeepResearch。你可以直接问我问题，也可以让我做深度研究。"
    if simple in {"thanks", "thank you", "thx"} or stripped in {"谢谢", "感谢"}:
        return "不客气。"
    if simple in {"bye", "goodbye"} or stripped in {"再见", "拜拜"}:
        return "再见，有需要随时继续。"
    if simple in {"who are you", "what can you do"} or stripped in {"你是谁", "你能做什么", "介绍一下你自己"}:
        return "我是 DeepResearch，一个本地运行的多 Agent 研究助手，可以做直答、联网调研、本地 RAG 检索、记忆和评测诊断。"
    if simple in {"what time", "what time is it", "time"} or stripped in {"现在几点", "时间"}:
        return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"
    if simple in {"date", "what date is it", "today"} or stripped in {"日期", "今天几号"}:
        return f"今天是：{datetime.now().strftime('%Y年%m月%d日')}"

    expr = _normalize_arithmetic_expression(stripped)
    if not expr:
        return None
    try:
        return _format_arithmetic_result(_eval_arithmetic_expression(expr))
    except Exception:
        return None


def _fallback_direct_answer(query: str, exc: Exception) -> str:
    q = query.strip().lower()
    if q.startswith(("hi", "hello", "hey")) or query.strip() in {"你好", "您好"}:
        return (
            "你好，我是 DeepResearch。当前模型服务连接不稳定，但我仍可以帮你做项目调研、"
            "Agent 评测诊断和报告生成；请稍后重试需要模型生成的请求。"
        )
    return (
        "模型服务暂时不可用，无法完成这次直接回答。\n\n"
        f"错误类型：{type(exc).__name__}\n"
        "建议检查本机到 dashscope.aliyuncs.com 的 DNS/HTTPS 连接、代理或网络策略后重试。"
    )


def direct_answer_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[direct_answer]", "cyan"), colorize(agent_name, "magenta"))
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S (星期%w)")
    prompt = f"当前时间：{now}\n用户问题：{state['query']}"
    human = HumanMessage(content=with_memory_context(state, prompt))
    # Include conversation history so multi-turn context is preserved
    content = _deterministic_direct_answer(state["query"])
    if content is not None:
        assistant_msg = AIMessage(content=content)
        audit_flags = []
    else:
        try:
            result = agent.invoke({"messages": state["messages"] + [human]})
            content = _last_content(result).strip()
            assistant_msg = result["messages"][-1]
            audit_flags = []
        except Exception as exc:
            logger.warning("direct_answer model call failed, using fallback: %s", exc)
            content = _fallback_direct_answer(state["query"], exc)
            assistant_msg = AIMessage(content=content)
            audit_flags = [
                {
                    "stage": "direct_answer",
                    "type": "model_error",
                    "message": str(exc),
                }
            ]
    emit("direct_answer", content)
    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "audit_flags": audit_flags,
        "messages": [human, assistant_msg],
    }



