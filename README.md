# 健康用药助手

这是一个面向中文健康用药场景的 RAG 问答项目。系统通过 FastAPI 提供接口，结合 DashScope 大模型、Redis 向量检索、BM25 稀疏检索、MySQL 药品数据查询和会话记忆，回答药物相互作用、饮食禁忌、个人用药相关问题。

## 功能概览

- 健康用药问答：支持非流式和 SSE 流式回复。
- 混合检索：Redis 向量检索 + BM25 稀疏检索，通过 RRF 融合排序。
- 查询理解：包含轻量意图识别、医学 NER、药品别名匹配和多轮上下文补全。
- 工具调用：可查询 MySQL 中的 `real_drug`、`yinshi` 等业务表。
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
- Redis Stack 或带 RediSearch 的 Redis：用于向量索引 `drug_vectors` 和会话记忆。
- MySQL：用于用户认证、药品库和个人用药数据查询。

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
```

MySQL 至少需要按代码使用到的表准备数据：

- `users(id, username, password_hash, created_at, updated_at)`：注册登录使用。
- `real_drug`：药品详情、饮食禁忌、相互作用等信息。
- `yinshi`：个人用药/饮食问题相关信息。

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

记忆摘要建议采用“短期窗口 + 长期摘要”的策略：

- 短期记忆：始终保留最近 7 轮原文，用于回答时保持上下文细节。
- 长期摘要：当对话超过 7 轮时，把更早的历史压缩成摘要，单独保存到类似 `chat_memory_summary:{memory_id}` 的 Redis key。
- 摘要内容只保留稳定信息，例如用户长期用药、已提到的疾病/过敏史、明确禁忌、偏好和已经澄清过的问题；不要保存一次性闲聊和不确定推测。
- 摘要更新采用增量方式：每次窗口外历史增加时，用旧摘要 + 新溢出的轮次生成新摘要，控制在 500-1000 字以内。
- 回答时的上下文顺序建议为：长期摘要、最近 7 轮对话、当前问题、RAG 检索内容。摘要只作为辅助上下文，专业结论仍应优先受知识库和数据库结果约束。

当前代码已经实现短期 7 轮记忆；长期摘要属于推荐策略入。

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

流式聊天：

```bash
curl -N "http://127.0.0.1:8000/chat-stream?memory_id=demo&message=布洛芬有什么饮食禁忌"
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

## 常见问题

- `缺少环境变量`：检查 `.env` 是否存在，变量名是否和 `config.py` 一致。
- Redis 向量索引失败：确认使用的是 Redis Stack，或 Redis 已启用 RediSearch。
- DashScope embedding 报批量错误：项目中已按 10 条一批处理，仍失败时检查模型名、API Key 和网络连通性。
- MySQL 查询失败：确认数据库、表结构和账号权限；数据库工具初始化失败不会阻断 RAG 启动，但相关工具调用会不可用。
