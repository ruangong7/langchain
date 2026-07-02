"""运行 RAGAS 评估"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *  # noqa: F403
from evaluation.ragas_evaluator import RAGASEvaluator
from langchain_community.vectorstores import Redis
from services.embedding_factory import build_embeddings
from services.llm_service import LLMService
from services.rag_service import RAGService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_services(top_k: int = 5, index_name: str = VECTOR_INDEX_NAME):  # noqa: F405
    """初始化 RAG 和 LLM 服务（仅稠密检索）"""
    logger.info("正在初始化服务...")

    # 初始化LLM服务
    llm_service = LLMService(tools=None)

    embeddings = build_embeddings()

    redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"  # noqa: F405
    vectorstore = Redis(
        redis_url=redis_url,
        index_name=index_name,
        embedding=embeddings,
    )

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    rag_service = RAGService(dense_retriever, title_index={})

    logger.info("服务初始化完成")
    return rag_service, llm_service, embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RAGAS 评估")
    parser.add_argument(
        "--file",
        default=os.path.join("evaluation", "test_data.generated.csv"),
        help="测试集文件路径（csv/xlsx）",
    )
    parser.add_argument(
        "--question-col",
        default="question",
        help="问题列名",
    )
    parser.add_argument(
        "--ground-truth-col",
        default="ground_truth",
        help="标准答案列名；若不存在将自动只评估 faithfulness/answer_relevance",
    )
    parser.add_argument(
        "--memory-id",
        default="evaluation_test",
        help="评估会话的 memory_id 前缀",
    )
    parser.add_argument(
        "--output",
        default="",
        help="评估结果输出路径（默认自动生成）",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="检索 Top-K，默认 5",
    )
    parser.add_argument(
        "--index-name",
        default=VECTOR_INDEX_NAME,  # noqa: F405
        help="Redis 向量索引名，默认 drug_vectors",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    # 初始化服务
    rag_service, llm_service, embeddings = init_services(top_k=args.top_k, index_name=args.index_name)

    # 创建评估器
    evaluator = RAGASEvaluator(rag_service, llm_service, eval_embeddings=embeddings)

    results = evaluator.evaluate_from_file(
        file_path=args.file,
        question_col=args.question_col,
        ground_truth_col=args.ground_truth_col,
        memory_id=args.memory_id,
    )

    # 保存结果
    output_path = evaluator.save_results(results, output_path=args.output or None)

    print(f"\n评估完成，结果文件：{output_path}")
    print("\n评估完成！")


if __name__ == "__main__":
    main()
