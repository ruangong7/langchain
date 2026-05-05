# RAGAS 评估体系

本目录包含使用 RAGAS（Retrieval-Augmented Generation Assessment）框架评估 RAG 系统的代码。

## 目录结构

```
evaluation/
├── __init__.py              # 模块初始化
├── ragas_evaluator.py       # RAGAS评估器主类
├── run_evaluation.py        # 运行评估的脚本
├── test_data.csv            # 测试数据集示例
└── README.md                # 本文件
```

## 安装依赖

首先需要安装 RAGAS 和相关依赖：

```bash
pip install ragas datasets pandas openpyxl
```

或者添加到 `requirements.txt`：

```
ragas>=0.1.0
datasets>=2.14.0
pandas>=1.5.0
openpyxl>=3.0.0
```

## 评估指标

RAGAS 提供以下评估指标：

1. **Context Precision（上下文精确度）**
   - 评估检索到的上下文是否与问题相关
   - 范围：0-1，越高越好

2. **Context Recall（上下文召回率）**
   - 评估是否检索到了所有相关的上下文
   - 范围：0-1，越高越好

3. **Faithfulness（忠实度）**
   - 评估答案是否基于提供的上下文生成
   - 范围：0-1，越高越好

4. **Answer Relevance（答案相关性）**
   - 评估答案是否与问题相关
   - 范围：0-1，越高越好

5. **Answer Correctness（答案正确性）**
   - 需要标准答案（ground truth）
   - 评估答案与标准答案的匹配程度
   - 范围：0-1，越高越好

## 使用方法

### 方法1：直接评估问题列表

```python
from evaluation.ragas_evaluator import RAGASEvaluator
from services.rag_service import RAGService
from services.llm_service import LLMService

# 初始化服务
rag_service = ...  # 你的RAG服务实例
llm_service = ...  # 你的LLM服务实例

# 创建评估器
evaluator = RAGASEvaluator(rag_service, llm_service)

# 定义测试问题
questions = [
    "阿莫西林的相互作用有哪些？",
    "哪些药物不能与阿司匹林一起服用？",
]

# 可选：标准答案
ground_truths = [
    "阿莫西林与多种药物存在相互作用...",
    "阿司匹林不能与抗凝药物一起服用...",
]

# 执行评估
results = evaluator.evaluate_rag(
    questions=questions,
    ground_truths=ground_truths,  # 可选
    memory_id="evaluation",
)

# 保存结果
evaluator.save_results(results)
```

### 方法2：从CSV文件读取问题

```python
# 从CSV文件读取问题并评估
results = evaluator.evaluate_from_file(
    file_path="evaluation/test_data.csv",
    question_col="question",
    ground_truth_col="ground_truth",  # 可选
    memory_id="evaluation",
)

# 保存结果
evaluator.save_results(results)
```

### 方法3：运行评估脚本

```bash
cd consultant_py
python evaluation/run_evaluation.py
```

## 测试数据格式

CSV文件格式示例（`test_data.csv`）：

```csv
question,ground_truth
阿莫西林的相互作用有哪些？,阿莫西林与多种药物存在相互作用...
哪些药物不能与阿司匹林一起服用？,阿司匹林不能与抗凝药物一起服用...
```

- `question`: 必填，测试问题
- `ground_truth`: 可选，标准答案（用于计算answer_correctness）

## 评估结果

评估结果会保存为CSV文件，包含以下列：

- `question`: 问题
- `contexts`: 检索到的上下文
- `answer`: RAG系统生成的答案
- `ground_truth`: 标准答案（如果有）
- `context_precision`: 上下文精确度分数
- `context_recall`: 上下文召回率分数
- `faithfulness`: 忠实度分数
- `answer_relevance`: 答案相关性分数
- `answer_correctness`: 答案正确性分数（如果有标准答案）

## 注意事项

1. **API调用成本**：RAGAS评估需要调用LLM API来计算某些指标，会产生API调用费用
2. **评估时间**：评估大量问题可能需要较长时间
3. **标准答案**：提供标准答案可以获得更全面的评估结果（包括answer_correctness）
4. **环境变量**：确保已正确配置 `DASHSCOPE_API_KEY` 等环境变量

## 参考

- [RAGAS官方文档](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
