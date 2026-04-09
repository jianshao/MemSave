import os
import sqlite3
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from .optimizer import get_model, get_vectorstore, extract_usage, builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

class MemSaveEngine:
    def __init__(self):
        self.llm = get_model()
        self.vectorstore = get_vectorstore()

    async def run(self, user_input: str, thread_id: str):
        # 统一输出逻辑，方便 API 或 MCP 调用
        # 使用异步上下文管理器连接数据库
        async with AsyncSqliteSaver.from_conn_string("storage/agent_memory.sqlite") as saver:
            app = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": thread_id}}
            result = await app.ainvoke({"input": user_input}, config=config)
            return result
