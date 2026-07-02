# 健康用药助手

这是一个面向中文健康用药场景的 RAG 问答项目。系统通过 FastAPI 提供接口，结合 DashScope 大模型、GraphRAG/检索组件、MySQL 用户健康档案和会话记忆，回答药物相互作用、用药风险和个人用药相关问题。

## 功能概览

- 健康用药问答：支持非流式和 SSE 流式回复。
- 混合检索：Redis 向量检索 + BM25 稀疏检索，通过 RRF 融合排序。
- 查询理解：包含轻量意图识别、医学 NER、药品别名匹配和多轮上下文补全。
- 工具调用：按登录用户读取健康档案与当前用药，为个体化问答补充背景。
- 会话记忆：支持短期历史、长期摘要、结构化长期记忆，以及会话历史读取/清空。
- 会话管理：支持游客/登录用户的多会话切换；登录用户的会话列表会同步到后端 Redis，可删除单个会话，并支持会话改名与最近时间展示。
- 用户认证：提供注册、登录接口，并返回本地签名 token。
- 评测脚本：包含召回评测和 RAGAS 评测入口。

## 目录结构

```text
.
├── main.py                         # FastAPI 应用入口
├── config.py                       # 环境变量配置
├── requirements.txt                # Python 依赖
├── static/                         # 前端静态页面
├── services/                       # LLM、RAG、记忆、认证、查询理解等服务
├── retriever/                      # 稀疏检索和混合检索实现
├── tools/                          # MySQL 查询工具
├── data_process/                   # 文档切分、向量写入 Redis 等脚本
├── content/                        # 知识库原始内容和分块数据
├── data/                           # 训练/评测种子数据、药品别名表
├── drug_kg/                        # 药品知识图谱与抽取相关脚本/结果
└── evaluation/                     # 召回评测与 RAGAS 评测脚本
```

## 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

项目依赖以下外部服务：

- DashScope：用于聊天模型和文本向量模型。
- Redis Stack 或带 RediSearch 的 Redis：用于会话记忆，以及按配置启用的旧检索组件。
- MySQL：用于用户认证、健康档案和个人用药数据存储。

## 配置

在项目根目录创建 `.env`，按需填写以下配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
EMBEDDING_MODEL=text-embedding-v2

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=your_database
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password

LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
CHAT_MEMORY_TTL_SECONDS=86400
CHAT_MEMORY_MAX_MESSAGES=20
CHAT_MEMORY_SUMMARY_BATCH_TURNS=3
CHAT_MEMORY_SUMMARY_TTL_SECONDS=2592000
AUTH_TOKEN_SECRET=change_me_to_a_long_random_secret
AUTH_TOKEN_TTL_SECONDS=86400

RAG_MAX_CONTEXT_CHARS=6000
RAG_MAX_DOC_CHARS=1200
RAG_FINAL_TOP_K=5
REWRITE_SHORT_QUERY_CHARS=12

DRUG_ALIAS_FILE=data/drug_aliases.csv
QUERY_UNDERSTANDING_ENABLE_LLM_FALLBACK=true
LIGHT_INTENT_MODEL_ENABLED=false
LIGHT_INTENT_MODEL_PATH=models/light_intent
LIGHT_INTENT_MODEL_MAX_LENGTH=128
MEDICAL_NER_ENABLED=false
MEDICAL_NER_MODEL_PATH=models/medical_ner
LLM_TOOL_CALLS_AVAILABLE=true
LLM_TOOL_CALLS_ENABLED=false
LLM_TOOL_CALLS_RUNTIME_OVERRIDE_ENABLED=true
```

MySQL 至少需要按代码使用到的表准备数据：

- `users(id, username, password_hash, created_at, updated_at)`：注册登录使用。
- `user_health_profile` / `user_medications`：用户健康档案和当前用药。

工具调用相关配置说明：

- `LLM_TOOL_CALLS_AVAILABLE`：是否在服务端注册工具 schema 和 handler。
- `LLM_TOOL_CALLS_ENABLED`：默认是否对请求开放工具调用。
- `LLM_TOOL_CALLS_RUNTIME_OVERRIDE_ENABLED`：是否允许前端或 API 按请求覆盖默认策略。

## 启动服务

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问：

- 前端页面：`http://127.0.0.1:8000/`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`

首次启动时，如果 Redis 中没有 `drug_vectors` 索引，应用会读取 `content/相互作用.md` 构建 BM25 文档，并尝试写入向量索引。

## 记忆策略

会话记忆按 `memory_id` 隔离，存储在 Redis 的 `chat_memory:{memory_id}` 列表中。每次回答完成后，系统会追加一条用户消息和一条助手消息，并按 `CHAT_MEMORY_TTL_SECONDS` 设置过期时间。

当前回答阶段采用最近 7 轮对话作为上下文：

- `services/llm_service.py` 中的 `ANSWER_HISTORY_TURNS = 7` 控制回答模型最多读取最近 7 轮历史。
- 1 轮对话包含 1 条用户消息和 1 条助手消息，因此至少需要保留 14 条消息。
- `CHAT_MEMORY_MAX_MESSAGES` 控制 Redis 中保留的消息条数，建议设置为 `14` 或更大；如果设置为 `20`，Redis 会保留最近 10 轮，但最终回答仍只取最近 7 轮。
- 查询理解中的上下文消歧会读取最近 5 轮，用于处理“这个药”“刚才那个”等含糊指代。

当前代码采用“短期窗口 + 长期摘要 + 用户档案缓存”的组合：

- 短期记忆：保留 Redis 列表 `chat_memory:{memory_id}` 中最近若干轮原文，用于回答时保持上下文细节。
- 长期摘要：窗口外历史按批次压缩到 `chat_memory_summary:{memory_id}`，默认累计超出 3 轮后再合并，保留稳定背景信息。
- 用户档案缓存：登录用户的个人档案会缓存到 Redis，减少每轮都回源 MySQL 的开销。
- TTL 拆分：`CHAT_MEMORY_TTL_SECONDS` 控制短期会话历史；`CHAT_MEMORY_SUMMARY_TTL_SECONDS` 控制长期摘要和用户档案缓存的保留时长。
- 会话管理：前端和接口支持按 `memory_id` 读取最近会话历史，并清空当前会话。
- 多会话兼容：登录用户默认沿用旧的 `user_id_{uid}` 会话，同时允许新建 `user_id_{uid}_session_*` 会话；游客会话也会在浏览器本地维护多个 `memory_id`。

## API 示例

注册：

```bash
curl -X POST http://127.0.0.1:8000/auth/register ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"demo\",\"password\":\"123456\"}"
```

登录：

```bash
curl -X POST http://127.0.0.1:8000/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"demo\",\"password\":\"123456\"}"
```

非流式聊天：

```bash
curl "http://127.0.0.1:8000/chat?memory_id=demo&message=阿司匹林不能和哪些药一起吃"
```

按请求实验性开启工具调用：

```bash
curl "http://127.0.0.1:8000/chat?memory_id=demo&message=我正在吃缬沙坦，还能吃布洛芬吗&include_meta=true&tool_policy=force_on"
```

流式聊天：

```bash
curl -N "http://127.0.0.1:8000/chat-stream?memory_id=demo&message=布洛芬有什么饮食禁忌"
```

读取会话历史：

```bash
curl "http://127.0.0.1:8000/chat-history?memory_id=demo&turns=20"
```

清空当前会话：

```bash
curl -X DELETE "http://127.0.0.1:8000/chat-history?memory_id=demo"
```

读取登录用户的会话列表：

```bash
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8000/me/chat-sessions"
```

查看系统运行时状态：

```bash
curl "http://127.0.0.1:8000/system/runtime-status"
```

## 数据处理

将 JSONL 分块数据写入 Redis 向量库：

```bash
python data_process/embed_chunks_to_redis.py ^
  --input content/gongzhonghao_text_chunks.jsonl ^
  --index-name drug_vectors ^
  --key-prefix qwen3:
```

脚本默认只向量化每行的 `text` 字段，并把 `source`、`source_file`、`chunk_index`、`chunk_count`、`chunk_id` 等字段写入 metadata。

## 评测

召回评测：

```bash
python evaluation/recall/run_hybrid_retrieval_recall.py
python evaluation/recall/run_bm25_retrieval_recall.py
python evaluation/recall/run_Denseretrieval_recall.py
```

RAGAS 评测：

```bash
python evaluation/ragas/run_evaluation.py
```

评测脚本会读取 `.env` 中的 DashScope、Redis 等配置；运行前请确认 Redis 中已有对应语料和索引。

## 回归测试

目前仓库已补充一组后端回归测试，覆盖：

- 会话结构化长期记忆的抽取与格式化
- ChatOrchestrator 的工具暴露策略
- 路由/检索元数据的基本行为
- 认证、健康档案读写、登录态校验
- 会话历史读取与清空接口
- 运行时状态接口

建议在项目环境中运行，例如：

```bash
/Users/david/miniconda3/envs/drug_agent/bin/python -m unittest discover -s tests -p "test_*.py"
```

## 常见问题

- `缺少环境变量`：检查 `.env` 是否存在，变量名是否和 `config.py` 一致。
- Redis 向量索引失败：确认使用的是 Redis Stack，或 Redis 已启用 RediSearch。
- DashScope embedding 报批量错误：项目中已按 10 条一批处理，仍失败时检查模型名、API Key 和网络连通性。
- MySQL 查询失败：确认数据库、表结构和账号权限；数据库工具初始化失败不会阻断 RAG 启动，但相关工具调用会不可用。
