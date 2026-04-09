import os
from mcp.server.fastmcp import FastMCP
from core.engine import MemSaveEngine

# 初始化 MCP
mcp = FastMCP("MemSave-Service")
# 初始化记忆引擎
engine = MemSaveEngine()

@mcp.tool()
async def chat_with_memory(user_input: str, user_id: str = "default_user") -> str:
    """
    【MemSave 核心能力】输入用户问题，自动调用长期记忆并优化 Token 消耗。
    """
    result = await engine.run(user_input, user_id)
    
    # 构造专业的回馈，体现服务价值
    stats = result.get("token_stats", {})
    saved = stats.get("baseline_p", 0) - stats.get("t", 0)
    
    response = (
        f"{result['generation']}\n\n"
        f"--- MemSave Insights ---\n"
        f"🧠 当前记忆摘要: {result['summary'][:50]}...\n"
        f"💰 本轮为你节省: {max(0, saved)} Tokens"
    )
    return response

# 修改 server_mcp.py 最末尾
if __name__ == "__main__":
    import asyncio
    import sys

    # 如果有命令行参数 test，则运行本地测试，否则运行 MCP 模式
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        config = {"configurable": {"thread_id": "global_user_1"}}
        print(f"模式: {os.getenv('LLM_PROVIDER')} | 模型: {os.getenv('LLM_MODEL')}")
        
        while True:
            user_in = input("\nUser > ")
            if user_in.lower() in ['q', 'exit']: break
            
            asyncio.run(chat_with_memory(user_in))

    else:
        mcp.run()
