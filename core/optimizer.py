import os
import sqlite3
from typing import List, TypedDict, Annotated, Dict, Any
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# --- 1. 模型工厂 (支持线上/本地混搭) ---
def get_model():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model_name = os.getenv("LLM_MODEL", "qwen2.5:7b")
    
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=0.1)
    elif provider == "deepseek":
        return ChatOpenAI(
            model=model_name,
            openai_api_base="https://deepseek.com",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.1
        )
    else: # 默认 ollama
        return ChatOllama(model=model_name, temperature=0.1)

def get_vectorstore():
    p = os.getenv("EMB_PROVIDER", "ollama")
    m = os.getenv("EMB_MODEL", "nomic-embed-text")
    
    if p == "openai":
        emb = OpenAIEmbeddings()
    else:
        emb = OllamaEmbeddings(model=m)
        
    path = f"../storage/db_v2_{p}_{m.replace(':', '_')}"
    return Chroma(persist_directory=path, embedding_function=emb, collection_name="long_term_v2")

llm = get_model()
vectorstore = get_vectorstore()

# --- 2. 状态定义 (保持不变) ---
class AgentState(TypedDict):
    input: str
    chat_history: Annotated[List[BaseMessage], "短期窗口"]
    summary: str
    context: str
    generation: str
    token_stats: Dict[str, int]

# --- 3. 增强版工具函数 ---
def extract_usage(response: Any) -> Dict[str, int]:
    """统一不同 Provider 的 Token 返回格式"""
    # 优先尝试 LangChain 统一格式
    usage = getattr(response, "usage_metadata", {}) or {}
    if not usage and hasattr(response, "additional_kwargs"):
        usage = response.additional_kwargs.get("token_usage", {})
        
    p = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    c = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return {"p": p, "c": c, "t": p + c}

# --- 4. 节点逻辑 (注入工厂模型) ---

def rewrite_and_retrieve_node(state: AgentState):
    summary = state.get("summary", "无")
    rewrite_p = f"基于对话摘要：{summary}\n将用户当前输入转化为1个精准检索词：{state['input']}"
    
    res = llm.invoke(rewrite_p)
    search_query = res.content
    
    docs = vectorstore.similarity_search(search_query, k=2)
    context = "\n".join([f"· {d.page_content}" for d in docs])
    
    return {
        "context": context,
        "token_stats": extract_usage(res)
    }

def generate_node(state: AgentState):
    short_history = state.get("chat_history", [])[-8:]
    current_summary = state.get("summary", "【事实】: 首次对话\n【偏好】: 未知")
    
    system_msg = f"记忆体：\n{current_summary}\n事实背景：\n{state['context']}\n请回答用户。"
    
    # 核心回答
    response = llm.invoke([SystemMessage(content=system_msg)] + short_history + [HumanMessage(content=state["input"])])
    
    # 摘要更新
    sum_update_p = f"更新摘要。新交互：用户说'{state['input']}'，AI说'{response.content}'\n旧摘要：{current_summary}\n输出更新后的结构化摘要："
    new_summary_res = llm.invoke(sum_update_p)
    
    # 计算 Baseline (字符数估算)
    history_chars = sum([len(m.content) for m in state.get("chat_history", [])])
    baseline_p = int((len(system_msg) + history_chars + len(state["input"])) * 0.8)

    u1, u2 = extract_usage(response), extract_usage(new_summary_res)
    
    return {
        "generation": response.content,
        "summary": new_summary_res.content,
        "chat_history": short_history + [HumanMessage(content=state["input"]), AIMessage(content=response.content)],
        "token_stats": {"t": u1["t"] + u2["t"], "baseline_p": baseline_p}
    }

def memorize_node(state: AgentState):
    p = f"提取核心硬事实（Key: Value格式，无则输出NONE）：\n{state['input']} | {state['generation']}"
    res = llm.invoke(p)
    
    if "NONE" not in res.content.upper() and len(res.content) > 5:
        vectorstore.add_texts([res.content.strip()])
    
    return {"token_stats": extract_usage(res)}

# --- 5. 流程编排 (保持不变) ---
builder = StateGraph(AgentState)
builder.add_node("retrieve", rewrite_and_retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("memorize", memorize_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "memorize")
builder.add_edge("memorize", END)


# --- 6. 运行控制 ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "global_user_1"}}
    print(f"模式: {os.getenv('LLM_PROVIDER')} | 模型: {os.getenv('LLM_MODEL')}")
    
    conn = sqlite3.connect("storage/agent_v2.sqlite", check_same_thread=False)
    app = builder.compile(checkpointer=SqliteSaver(conn))
    while True:
        user_in = input("\nUser > ")
        if user_in.lower() in ['q', 'exit']: break
        
        total_actual = 0
        baseline = 0
        
        for event in app.stream({"input": user_in}, config=config, stream_mode="updates"):
            for node, output in event.items():
                if "token_stats" in output:
                    total_actual += output["token_stats"].get("t", 0)
                    if "baseline_p" in output["token_stats"]: 
                        baseline = output["token_stats"]["baseline_p"]
                if node == "generate":
                    print(f"AI > {output['generation']}")

        print(f"\033[32m[Token 节省] 本轮实耗: {total_actual} | 预估节省: {max(0, baseline-total_actual)}\033[0m")
