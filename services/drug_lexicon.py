"""药品词表服务。"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from config import DRUG_ALIAS_FILE
from tools.database_tool import DatabaseTool

logger = logging.getLogger(__name__)


class DrugLexiconService:
    """从数据库和本地扩展文件加载药品名称、别名、商品名词表。"""

    def __init__(self, database_tool: Optional[DatabaseTool]):
        self.database_tool = database_tool
        self._lexicon: Dict[str, str] = {}

    def load(self, refresh: bool = False) -> Dict[str, str]:
        if self._lexicon and not refresh:
            return self._lexicon
        alias_map: Dict[str, str] = {}
        if self.database_tool is None:
            logger.warning("数据库工具不可用，仅加载本地药品别名文件")
        else:
            for name in self.database_tool.load_drug_lexicon(refresh=refresh):
                self._add_alias(alias_map, name, name)

        for alias, canonical in self._load_alias_file(DRUG_ALIAS_FILE).items():
            self._add_alias(alias_map, alias, canonical)

        self._lexicon = dict(sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True))
        logger.info(
            "药品词表加载完成: aliases=%d canonical=%d",
            len(self._lexicon),
            len(set(self._lexicon.values())),
        )
        return self._lexicon

    def match_mentions(self, text: str, refresh: bool = False) -> List[Dict[str, str]]:
        lexicon = self.load(refresh=refresh)
        normalized_text = " ".join(str(text or "").strip().split())
        if not normalized_text or not lexicon:
            return []

        matches: List[Dict[str, str]] = []
        occupied_ranges: List[tuple[int, int]] = []
        for alias, canonical in lexicon.items():
            start = normalized_text.find(alias)
            if start < 0:
                continue
            end = start + len(alias)
            if any(not (end <= left or start >= right) for left, right in occupied_ranges):
                continue
            occupied_ranges.append((start, end))
            matches.append(
                {
                    "mention": alias,
                    "canonical": canonical,
                }
            )
        return matches

    def _load_alias_file(self, file_path: str) -> Dict[str, str]:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            logger.warning("药品别名文件不存在: %s", path)
            return {}

        alias_map: Dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    canonical = str(row.get("canonical") or row.get("通用名") or "").strip()
                    aliases = str(row.get("aliases") or row.get("别名") or "").strip()
                    if not canonical:
                        continue
                    self._add_alias(alias_map, canonical, canonical)
                    for alias in aliases.replace("，", ",").replace("、", ",").split(","):
                        self._add_alias(alias_map, alias, canonical)
            logger.info("药品别名文件加载完成: %s aliases=%d", path, len(alias_map))
        except Exception as exc:
            logger.warning("药品别名文件加载失败: %s error=%s", path, exc, exc_info=True)
        return alias_map

    @staticmethod
    def _add_alias(alias_map: Dict[str, str], alias: str, canonical: str) -> None:
        alias = " ".join(str(alias or "").strip().split())
        canonical = " ".join(str(canonical or "").strip().split())
        if alias and canonical:
            alias_map[alias] = canonical
