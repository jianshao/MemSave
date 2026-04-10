import os
import sqlite3
from typing import List, TypedDict, Dict, Any
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# --- 1. 模型工厂 ---
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
    return ChatOllama(model=model_name, temperature=0.1)

def get_vectorstore():
    p = os.getenv("EMB_PROVIDER", "ollama")
    m = os.getenv("EMB_MODEL", "nomic-embed-text")
    emb = OpenAIEmbeddings() if p == "openai" else OllamaEmbeddings(model=m)
    path = f"./storage/db_v2_{p}_{m.replace(':', '_')}"
    return Chroma(persist_directory=path, embedding_function=emb, collection_name="long_term_v2")

llm = get_model()
vectorstore = get_vectorstore()

# --- 2. 状态定义 ---
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    summary: str
    context: str
    generation: str
    token_stats: Dict[str, int]
    # 新增：持久化统计
    total_actual_tokens: int 
    total_baseline_tokens: int

# --- 3. 工具函数 ---
def extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", {}) or {}
    if not usage and hasattr(response, "additional_kwargs"):
        usage = response.additional_kwargs.get("token_usage", {})
    p = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    c = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return {"p": p, "c": c, "t": p + c}

# --- 4. 节点逻辑 ---

def retrieve_node(state: AgentState):
    summary = state.get("summary", "无")
    if len(state['input']) > 20:
        rewrite_p = f"根据摘要：{summary}\n提取用户输入中的核心检索词（5词以内）：{state['input']}"
        res = llm.invoke(rewrite_p)
        query = res.content
        usage = extract_usage(res)
    else:
        query = state['input']
        usage = {"p": 0, "c": 0, "t": 0}

    docs = vectorstore.similarity_search(query, k=2)
    context = "\n".join([f"· {d.page_content}" for d in docs])
    return {"context": context, "token_stats": usage}

def generate_and_summarize_node(state: AgentState):
    short_history = state.get("chat_history", [])[-6:]
    current_summary = state.get("summary", "【事实】: 首次对话\n【偏好】: 未知")
    
    system_msg = f"你是一个高效助手。\n记忆体：{current_summary}\n背景：{state['context']}\n请回答用户。然后在末尾增加一行 '---UPDATE---' 紧跟更新后的结构化摘要。"

    response = llm.invoke([SystemMessage(content=system_msg)] + short_history + [HumanMessage(content=state["input"])])
    
    full_content = response.content
    if "---UPDATE---" in full_content:
        generation, new_summary = full_content.split("---UPDATE---", 1)
    else:
        generation, new_summary = full_content, current_summary

    # 计算 Baseline (全量历史模式)
    all_history_text = "".join([m.content for m in state.get("chat_history", [])])
    # 模拟在全量历史下，单次生成的 Prompt Token 消耗
    current_baseline_p = int((len(all_history_text) + len(state['input']) + 200) * 0.8)
    
    usage = extract_usage(response)
    
    # 累加统计逻辑
    current_total_actual = state.get("total_actual_tokens", 0) + usage["t"]
    current_total_baseline = state.get("total_baseline_tokens", 0) + current_baseline_p

    return {
        "generation": generation.strip(),
        "summary": new_summary.strip(),
        "chat_history": short_history + [HumanMessage(content=state["input"]), AIMessage(content=generation.strip())],
        "token_stats": {"t": usage["t"], "baseline_p": current_baseline_p},
        "total_actual_tokens": current_total_actual,
        "total_baseline_tokens": current_total_baseline
    }

def memorize_node(state: AgentState):
    p = f"提取此轮对话新事实（无则报NONE）：\nU: {state['input']}\nA: {state['generation']}"
    res = llm.invoke(p)
    usage = extract_usage(res)
    
    if "NONE" not in res.content.upper() and len(res.content) > 5:
        vectorstore.add_texts([res.content.strip()])
    
    # 将此节点的消耗也累加进总实耗
    return {
        "token_stats": usage,
        "total_actual_tokens": state.get("total_actual_tokens", 0) + usage["t"]
    }

# --- 5. 编排 ---
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_and_summarize_node)
builder.add_node("memorize", memorize_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "memorize")
builder.add_edge("memorize", END)

# --- 6. 运行控制 ---
if __name__ == "__main__":
    if not os.path.exists("storage"): os.makedirs("storage")
    conn = sqlite3.connect("storage/agent_v2.sqlite", check_same_thread=False)
    app = builder.compile(checkpointer=SqliteSaver(conn))
    
    config = {"configurable": {"thread_id": "stats_test_001"}}
    print(f"🚀 MemSave 模式启动 (含全量累积统计)")

    while True:
        user_in = input("\nUser > ")
        if user_in.lower() in ['q', 'exit']: break
        
        last_state = None
        for event in app.stream({"input": user_in}, config=config, stream_mode="updates"):
            for node, output in event.items():
                last_state = output # 捕捉最后一个节点的输出状态
                if node == "generate":
                    print(f"\nAI > {output['generation']}")

        # 从持久化状态中提取累积数据
        # 注意：由于 stream 返回的是 update，累积值在最终状态中
        final_snapshot = app.get_state(config).values
        t_actual = final_snapshot.get("total_actual_tokens", 0)
        t_baseline = final_snapshot.get("total_baseline_tokens", 0)
        t_saved = t_baseline - t_actual
        
        # 计算本轮数据（用于单轮对比）
        cur_actual = sum([v.get("t", 0) for k, v in last_state.items() if k == "token_stats"] or [0]) # 简化处理

        print("-" * 40)
        print(f"📊 [本轮实耗]: {last_state.get('token_stats', {}).get('t', '计算中...')} Tokens")
        print(f"📈 [累积统计] 实耗: {t_actual} | 预估全量模式: {t_baseline}")
        if t_saved > 0:
            print(f"\033[32m💰 累计节省: {t_saved} Tokens (约节省 {(t_saved/t_baseline*100):.1f}%)\033[0m")
        else:
            print(f"\033[33m⏳ 记忆构建中，节省效果随对话轮数增加而增强...\033[0m")
        print("-" * 40)
