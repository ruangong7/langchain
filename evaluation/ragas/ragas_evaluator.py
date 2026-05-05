"""使用RAGAS评估RAG系统"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional, Any
import pandas as pd
import logging
from datetime import datetime

from ragas import evaluate
from ragas import metrics as ragas_metrics
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.embeddings import Embeddings

from services.rag_service import RAGService
from services.llm_service import LLMService
from config import *

logger = logging.getLogger(__name__)


def _save_rag_checkpoint_to_disk(data: Dict[str, Any], eval_dir: str) -> str:
    """在调用 RAGAS 之前落盘检索与回答，避免判分阶段失败时丢失已跑完的样本。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, f"rag_predictions_{ts}.csv")
    n = len(data["question"])
    rows = []
    for i in range(n):
        ctx_cell = data["contexts"][i]
        ctx_text = ctx_cell[0] if ctx_cell else ""
        row = {
            "question": data["question"][i],
            "answer": data["answer"][i],
            "context": ctx_text,
        }
        if "ground_truth" in data:
            row["ground_truth"] = data["ground_truth"][i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("已保存 RAG 中间结果（检索+回答），判分失败也可从此文件恢复: %s", os.path.abspath(path))
    return path

# 兼容不同 RAGAS 版本的指标命名
context_precision = getattr(ragas_metrics, "context_precision")
context_recall = getattr(ragas_metrics, "context_recall")
faithfulness = getattr(ragas_metrics, "faithfulness")
answer_correctness = getattr(ragas_metrics, "answer_correctness")
answer_relevance = getattr(
    ragas_metrics,
    "answer_relevance",
    getattr(ragas_metrics, "answer_relevancy"),
)


class RAGASEvaluator:
    """使用RAGAS评估RAG系统"""
    
    def __init__(
        self,
        rag_service: RAGService,
        llm_service: LLMService,
        eval_embeddings: Optional[Embeddings] = None,
    ):
        """
        初始化评估器
        
        Args:
            rag_service: RAG服务实例
            llm_service: LLM服务实例
        """
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.eval_embeddings = eval_embeddings

        # RAGAS 在部分版本会内部直接初始化 OpenAI 客户端，
        # 这里将 DashScope 兼容配置映射为 OpenAI 环境变量，避免评估阶段报缺少 api_key。
        if DASHSCOPE_API_KEY:
            os.environ["OPENAI_API_KEY"] = DASHSCOPE_API_KEY
        if DASHSCOPE_BASE_URL:
            os.environ["OPENAI_BASE_URL"] = DASHSCOPE_BASE_URL
        
        # RAGAS 各指标会并发调 LLM；DashScope 在高压下可能返回空 JSON，导致 faithfulness 等解析失败。
        self._run_config = RunConfig(max_workers=2, timeout=300)
        # 用于 RAGAS 判分（结构化输出对温度敏感，尽量确定性）
        self.eval_llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base=DASHSCOPE_BASE_URL,
            temperature=0,
        )
    
    def evaluate_rag(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        memory_id: str = "evaluation",
    ) -> pd.DataFrame:
        """
        评估RAG系统
        
        Args:
            questions: 问题列表
            ground_truths: 标准答案列表（可选，用于计算answer_correctness）
            memory_id: 记忆ID，用于LLM服务
        
        Returns:
            评估结果DataFrame
        """
        logger.info(f"开始评估 {len(questions)} 个问题...")
        
        # 收集RAG系统的输出
        contexts = []
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")
            
            # 获取上下文
            context = self.rag_service.retrieve_context(question)
            contexts.append(context)
            
            # 获取答案
            answer = self.llm_service.chat(memory_id=f"{memory_id}_{i}", message=question, context=context)
            answers.append(answer)
        
        # 构建评估数据集
        data = {
            "question": questions,
            "contexts": [[ctx] for ctx in contexts],  # RAGAS需要contexts是列表的列表
            "answer": answers,
        }
        
        # 如果有标准答案，添加ground_truth
        if ground_truths:
            data["ground_truth"] = ground_truths

        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        _save_rag_checkpoint_to_disk(data, checkpoint_dir)
        
        dataset = Dataset.from_dict(data)
        
        # 定义评估指标
        # 注意：context_precision 和 context_recall 需要 ground_truth
        # 如果没有 ground_truth，只能评估 faithfulness 和 answer_relevance
        if ground_truths:
            metrics = [
                context_precision,  # 上下文精确度：检索到的上下文是否相关（需要ground_truth）
                context_recall,     # 上下文召回率：是否检索到所有相关上下文（需要ground_truth）
                faithfulness,        # 忠实度：答案是否基于上下文生成
                answer_relevance,    # 答案相关性：答案是否与问题相关
                answer_correctness,  # 答案正确性：答案与标准答案的匹配程度（需要ground_truth）
            ]
        else:
            # 没有标准答案时，只能评估忠实度和相关性
            metrics = [
                faithfulness,        # 忠实度：答案是否基于上下文生成
                answer_relevance,    # 答案相关性：答案是否与问题相关
            ]
            logger.info("未提供标准答案，将只评估忠实度（faithfulness）和答案相关性（answer_relevance）")
        
        # 执行评估
        logger.info("开始执行RAGAS评估...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=LangchainLLMWrapper(self.eval_llm, run_config=self._run_config),
            embeddings=(
                LangchainEmbeddingsWrapper(self.eval_embeddings)
                if self.eval_embeddings is not None
                else None
            ),
            run_config=self._run_config,
            raise_exceptions=False,
        )

        
        # 兼容不同 RAGAS 版本的结果结构
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
        else:
            df = pd.DataFrame(result)
        
        logger.info("评估完成")
        return df
    
    def evaluate_without_ground_truth(
        self,
        questions: List[str],
        memory_id: str = "evaluation",
    ) -> pd.DataFrame:
        """
        在没有标准答案的情况下评估RAG系统
        只评估忠实度（faithfulness）和答案相关性（answer_relevance）
        
        Args:
            questions: 问题列表
            memory_id: 记忆ID，用于LLM服务
        
        Returns:
            评估结果DataFrame（只包含faithfulness和answer_relevance）
        """
        logger.info(f"开始评估 {len(questions)} 个问题（无标准答案模式）...")
        logger.info("将只评估：忠实度（faithfulness）和答案相关性（answer_relevance）")
        
        # 收集RAG系统的输出
        contexts = []
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")
            
            # 获取上下文
            context = self.rag_service.retrieve_context(question)
            contexts.append(context)
            
            # 获取答案
            answer = self.llm_service.chat(memory_id=f"{memory_id}_{i}", message=question, context=context)
            answers.append(answer)
        
        # 构建评估数据集（只需要question, contexts, answer）
        data = {
            "question": questions,
            "contexts": [[ctx] for ctx in contexts],  # RAGAS需要contexts是列表的列表
            "answer": answers,
        }

        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        _save_rag_checkpoint_to_disk(data, checkpoint_dir)
        
        dataset = Dataset.from_dict(data)
        
        # 只评估不需要ground_truth的指标
        metrics = [
            faithfulness,        # 忠实度：答案是否基于上下文生成
            answer_relevance,    # 答案相关性：答案是否与问题相关
        ]
        
        # 执行评估
        logger.info("开始执行RAGAS评估（仅忠实度和相关性）...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=LangchainLLMWrapper(self.eval_llm, run_config=self._run_config),
            embeddings=(
                LangchainEmbeddingsWrapper(self.eval_embeddings)
                if self.eval_embeddings is not None
                else None
            ),
            run_config=self._run_config,
            raise_exceptions=False,
        )
        
        # 兼容不同 RAGAS 版本的结果结构
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
        else:
            df = pd.DataFrame(result)
        
        logger.info("评估完成")
        return df
    
    def evaluate_from_file(
        self,
        file_path: str,
        question_col: str = "question",
        ground_truth_col: Optional[str] = None,
        memory_id: str = "evaluation",
    ) -> pd.DataFrame:
        """
        从文件读取问题并评估
        
        Args:
            file_path: CSV或Excel文件路径
            question_col: 问题列名
            ground_truth_col: 标准答案列名（可选）
            memory_id: 记忆ID
        
        Returns:
            评估结果DataFrame
        """
        # 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        questions = df[question_col].tolist()
        ground_truths = None
        if ground_truth_col and ground_truth_col in df.columns:
            ground_truths = df[ground_truth_col].tolist()
        
        # 如果没有标准答案，使用无标准答案模式
        if ground_truths is None or len(ground_truths) == 0:
            logger.info("检测到没有标准答案，使用无标准答案评估模式（仅评估忠实度和相关性）")
            return self.evaluate_without_ground_truth(questions, memory_id)
        else:
            return self.evaluate_rag(questions, ground_truths, memory_id)
    
    def save_results(self, results_df: pd.DataFrame, output_path: str = None):
        """
        保存评估结果
        
        Args:
            results_df: 评估结果DataFrame
            output_path: 输出文件路径（默认：evaluation/results_YYYYMMDD_HHMMSS.csv）
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation/results_{timestamp}.csv"
        
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"评估结果已保存到: {output_path}")
        
        # 打印摘要统计
        print("\n=== RAGAS评估结果摘要 ===")
        print(f"总问题数: {len(results_df)}")
        print(f"\n平均分数:")
        excluded_cols = {
            "question",
            "contexts",
            "answer",
            "ground_truth",
            "user_input",
            "retrieved_contexts",
            "response",
            "reference",
        }
        printed = False
        for col in results_df.columns:
            if col in excluded_cols:
                continue
            numeric_col = pd.to_numeric(results_df[col], errors="coerce")
            if numeric_col.notna().sum() == 0:
                continue
            avg_score = numeric_col.mean()
            print(f"  {col}: {avg_score:.4f}")
            printed = True
        if not printed:
            print("  （未检测到可统计的数值指标列）")
        
        return output_path


if __name__ == "__main__":
    # 示例用法
    print("RAGAS评估器")
    print("请先初始化RAG服务和LLM服务，然后调用evaluate_rag方法")
