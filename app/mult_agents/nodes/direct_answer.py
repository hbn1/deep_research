import json, logging
from langchain_core.messages import HumanMessage
from ..state import ResearchState
from ..utils import (colorize, emit, collect_tool_calls, with_memory_context, log_inputs, _last_content)

logger = logging.getLogger('mult_agents')

def direct_answer_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[direct_answer]", "cyan"), colorize(agent_name, "magenta"))
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S (星期%w)")
    prompt = f"当前时间：{now}\n用户问题：{state['query']}"
    human = HumanMessage(content=with_memory_context(state, prompt))
    # Include conversation history so multi-turn context is preserved
    result = agent.invoke({"messages": state["messages"] + [human]})
    content = _last_content(result).strip()
    emit("direct_answer", content)
    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "messages": [human, result["messages"][-1]],
    }



