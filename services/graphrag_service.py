"""Official GraphRAG service wrapper for the medical QA pipeline."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import graphrag.api as graphrag_api
import pandas as pd
from dotenv import load_dotenv
from graphrag.config.load_config import load_config
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider
from graphrag.data_model.data_reader import DataReader

logger = logging.getLogger(__name__)

GLOBAL_QUERY_HINTS = (
    "最常见",
    "总体",
    "整体",
    "全局",
    "模式",
    "规律",
    "趋势",
    "总结",
    "概览",
    "高风险相互作用模式",
)


class GraphRAGService:
    """Use official GraphRAG local/global search as the retrieval backbone."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
        dynamic_community_selection: bool = False,
        max_context_chars: int = 12000,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.community_level = community_level
        self.response_type = response_type
        self.dynamic_community_selection = dynamic_community_selection
        self.max_context_chars = max_context_chars
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graphrag-query")

        load_dotenv(self.project_root / ".env")
        self.config = load_config(root_dir=self.project_root)
        self._tables = self._load_tables()
        logger.info(
            "GraphRAG 服务初始化完成: root=%s community_level=%s",
            self.project_root,
            self.community_level,
        )

    def retrieve_context_for_analysis(self, analysis: Dict[str, Any], fallback_query: str) -> str:
        query = str(analysis.get("resolved_query") or analysis.get("normalized_query") or fallback_query or "").strip()
        if not query:
            return ""

        method = self._pick_method(analysis, query)
        logger.info("GraphRAG 检索开始: method=%s intent=%s query=%s", method, analysis.get("intent"), query)

        if method == "global":
            response, context_data = self._run_global_sync(query)
        else:
            response, context_data = self._run_local_sync(query)

        self._log_debug_hits(method=method, query=query, context_data=context_data, response=response)
        context = self._format_context(method=method, query=query, response=response, context_data=context_data)
        logger.info(
            "GraphRAG 检索完成: method=%s context长度=%d query=%s",
            method,
            len(context),
            query,
        )
        return context

    def _pick_method(self, analysis: Dict[str, Any], query: str) -> str:
        intent = str(analysis.get("intent") or "").strip()
        compact = "".join(query.split())
        if intent in {"interaction", "contraindication", "side_effect", "dosage", "population", "drug_info"}:
            return "local"
        if any(hint in compact for hint in GLOBAL_QUERY_HINTS):
            return "global"
        return "local"

    def _load_tables(self) -> Dict[str, Optional[pd.DataFrame]]:
        storage = create_storage(self.config.output_storage)
        table_provider = create_table_provider(self.config.table_provider, storage=storage)
        reader = DataReader(table_provider)

        tables: Dict[str, Optional[pd.DataFrame]] = {
            "entities": self._run_async(async_fn=reader.entities),
            "communities": self._run_async(async_fn=reader.communities),
            "community_reports": self._run_async(async_fn=reader.community_reports),
            "text_units": self._run_async(async_fn=reader.text_units),
            "relationships": self._run_async(async_fn=reader.relationships),
        }
        has_covariates = self._run_async_value(table_provider.has("covariates"))
        tables["covariates"] = self._run_async(async_fn=reader.covariates) if has_covariates else None
        return tables

    def _run_local_sync(self, query: str) -> tuple[str, Dict[str, Any]]:
        return self._run_coro_sync(
            graphrag_api.local_search(
                config=self.config,
                entities=self._tables["entities"],
                communities=self._tables["communities"],
                community_reports=self._tables["community_reports"],
                text_units=self._tables["text_units"],
                relationships=self._tables["relationships"],
                covariates=self._tables["covariates"],
                community_level=self.community_level,
                response_type=self.response_type,
                query=query,
                verbose=False,
            )
        )

    def _run_global_sync(self, query: str) -> tuple[str, Dict[str, Any]]:
        return self._run_coro_sync(
            graphrag_api.global_search(
                config=self.config,
                entities=self._tables["entities"],
                communities=self._tables["communities"],
                community_reports=self._tables["community_reports"],
                community_level=self.community_level,
                dynamic_community_selection=self.dynamic_community_selection,
                response_type=self.response_type,
                query=query,
                verbose=False,
            )
        )

    def _run_coro_sync(self, coro):
        future = self._executor.submit(asyncio.run, coro)
        return future.result()

    def _run_async(self, *, async_fn):
        return self._run_async_value(async_fn())

    def _run_async_value(self, awaitable):
        future = self._executor.submit(asyncio.run, awaitable)
        return future.result()

    def _format_context(
        self,
        *,
        method: str,
        query: str,
        response: str,
        context_data: Dict[str, Any],
    ) -> str:
        parts = [
            f"【GraphRAG模式】{method}",
            f"【GraphRAG查询】{query}",
            f"【GraphRAG回答】{self._normalize_text(response)}",
        ]

        if method == "local":
            parts.extend(
                [
                    self._format_dataframe(
                        context_data.get("reports"),
                        label="GraphRAG社区报告",
                        preferred_columns=("title", "content", "rank"),
                        limit=2,
                    ),
                    self._format_dataframe(
                        context_data.get("entities"),
                        label="GraphRAG实体",
                        preferred_columns=("entity", "description", "number of relationships"),
                        limit=8,
                    ),
                    self._format_dataframe(
                        context_data.get("relationships"),
                        label="GraphRAG关系",
                        preferred_columns=("source", "target", "description"),
                        limit=8,
                    ),
                    self._format_dataframe(
                        context_data.get("sources"),
                        label="GraphRAG原文证据",
                        preferred_columns=("text",),
                        limit=4,
                    ),
                ]
            )
        else:
            parts.append(
                self._format_dataframe(
                    context_data.get("reports"),
                    label="GraphRAG社区报告",
                    preferred_columns=("title", "content", "rank"),
                    limit=6,
                )
            )

        context = "\n\n".join(part for part in parts if part.strip())
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "..."
        return context

    def _log_debug_hits(
        self,
        *,
        method: str,
        query: str,
        context_data: Dict[str, Any],
        response: str,
    ) -> None:
        sections = []
        if method == "local":
            sections.extend(
                [
                    self._summarize_dataframe(
                        context_data.get("entities"),
                        label="entities",
                        preferred_columns=("entity", "description", "number of relationships"),
                        limit=5,
                    ),
                    self._summarize_dataframe(
                        context_data.get("relationships"),
                        label="relationships",
                        preferred_columns=("source", "target", "description"),
                        limit=5,
                    ),
                    self._summarize_dataframe(
                        context_data.get("sources"),
                        label="sources",
                        preferred_columns=("text",),
                        limit=3,
                    ),
                    self._summarize_dataframe(
                        context_data.get("reports"),
                        label="reports",
                        preferred_columns=("title", "content", "rank"),
                        limit=2,
                    ),
                ]
            )
        else:
            sections.append(
                self._summarize_dataframe(
                    context_data.get("reports"),
                    label="reports",
                    preferred_columns=("title", "content", "rank"),
                    limit=5,
                )
            )

        response_preview = self._normalize_text(response)[:1000]
        joined = "\n".join(part for part in sections if part)
        logger.info(
            "GraphRAG 命中详情: method=%s query=%s\n%s\n[response_preview] %s",
            method,
            query,
            joined,
            response_preview,
        )

    def _format_dataframe(
        self,
        df: Any,
        *,
        label: str,
        preferred_columns: tuple[str, ...],
        limit: int,
    ) -> str:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return ""

        items = []
        for idx, row in enumerate(df.head(limit).to_dict(orient="records"), start=1):
            row_parts = []
            for col in preferred_columns:
                value = row.get(col)
                if value is None or value == "":
                    continue
                normalized = self._normalize_text(value)
                row_parts.append(f"{col}={normalized}")
            if row_parts:
                items.append(f"【{label}{idx}】" + "；".join(row_parts))
        return "\n".join(items)

    def _summarize_dataframe(
        self,
        df: Any,
        *,
        label: str,
        preferred_columns: tuple[str, ...],
        limit: int,
    ) -> str:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return f"[{label}] empty"

        lines = [f"[{label}] rows={len(df)}"]
        for idx, row in enumerate(df.head(limit).to_dict(orient="records"), start=1):
            parts = []
            for col in preferred_columns:
                value = row.get(col)
                if value is None or value == "":
                    continue
                parts.append(f"{col}={self._normalize_text(value)[:500]}")
            if parts:
                lines.append(f"  - {idx}. " + " | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = " ".join(str(value or "").split())
        return text[:2400]
