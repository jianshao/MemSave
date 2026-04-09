# 🧠 MemSave: 工业级 Agent 长期记忆与 Token 优化中枢

[![MCP](https://shields.io)](https://modelcontextprotocol.io)
[![License](https://shields.io)](LICENSE)
[![Token Savings](https://shields.io)](#-性能基准)

**MemSave** 是一款基于 **Model Context Protocol (MCP)** 的高性能 Agent 记忆管理服务。它不只是一个存储工具，而是一个自带逻辑闭环的“第二大脑”。通过独创的**三轨引擎（Triple-Track Engine）**，MemSave 能让你的 Agent 在跨越数月的对话中依然保持精准记忆，同时将 Token 成本降低 80% 以上。

---

## ✨ 核心价值：为什么要使用 MemSave？

传统的记忆方案只是简单的“全文存储”，会导致 Context 迅速膨胀。MemSave 通过以下机制重新定义 Agent 记忆：

- **🛡️ 三轨自适应引擎**：
  - **检索轨 (Retrieve)**：自动将模糊输入重写为精准关键词，实现秒级海量事实调取。
  - **摘要轨 (Summary)**：动态维护【事实/偏好/进度】结构化视图，确保 Agent 时刻清醒。
  - **沉淀轨 (Store)**：自动识别并持久化对话中的“硬事实”，无需人工干预。
- **💸 极致成本优化**：采用“状态压缩”算法，将 Token 消耗从 O(N) 降低到接近恒定的 O(1)，大幅削减 API 账单。
- **🔌 即插即用**：完美适配 Claude Desktop, Cursor, IDE 插件及任何支持 MCP 协议的 Agent 平台。
- **🔒 隐私级安全**：支持 Ollama 全本地部署，记忆体完全存储在本地 SQLite 和 Chroma 中。

---

## 📊 性能基准 (Benchmark)

在连续 50 轮的复杂开发对话测试中：


| 指标 | 原生 Agent 模式 | MemSave 驱动模式 | 优化率 |
| :--- | :--- | :--- | :--- |
| **单次请求平均 Token** | 24,500+ | **1,200 - 1,800** | **↘ 92%** |
| **长期事实召回率** | < 40% (随上下文丢失) | **> 98%** | **↗ 145%** |
| **单轮成本 (GPT-4o)** | ~$0.12 | **~$0.006** | **节省 95%** |

---

## 🚀 快速启动

### 1. 克隆并安装环境
```bash
git clone https://github.com/jianshao/MemSave.git
cd MemSave
pip install -r requirements.txt
```

### 2. 配置环境变量
复制 .env.example 到 .env 并填写你的配置：
```
# 模型供应商: ollama / openai / deepseek
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b

# 嵌入模型: 建议本地运行以确保隐私
EMB_PROVIDER=ollama
EMB_MODEL=nomic-embed-text
```

### 3. 接入 Claude Desktop
打开你的 claude_desktop_config.json，添加以下内容：
```
{
  "mcpServers": {
    "memsave_service": {
      "command": "python",
      "args": ["/你的绝对路径/MemSave/server_mcp.py"]
    }
  }
}
```


**🛠️ 进阶用法**

多租户/多项目管理
你可以通过指定 user_id 为不同的项目创建隔离的记忆空间：
user_id="project_alpha": 存储 A 项目的技术栈和决策记录。
user_id="personal_fin": 存储个人理财偏好和历史账单。

**核心 API 说明**

MemSave 暴露了一个高阶工具接口：
chat_with_memory(user_input, user_id):
输入：原始用户提问。
输出：经过背景增强的回答 + 结构化摘要更新 + 本轮节省统计。

**🗺️ 路线图 (Roadmap)**

基于 MCP 的三轨记忆闭环
多 Provider 模型工厂 (OpenAI/Ollama/DeepSeek)
Web UI 可视化面板：直观查看和编辑 Agent 的记忆事实
多端同步：支持云端加密备份
知识图谱升级：从向量检索升级为实体关联检索

**📄 开源协议**
本项目采用 Apache-2.0 协议。欢迎集成到你的商业应用中，但请保留原作者署名。
立即部署 MemSave，让你的 Agent 摆脱“金鱼脑”，开始享受廉价且强大的长期记忆。

⭐ Star Us on GitHub | 💬 加入讨论
