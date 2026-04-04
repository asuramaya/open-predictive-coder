from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decepticons import (  # noqa: E402
    OracleAnalysisAdapter as _OracleAnalysisAdapter,
    OracleAnalysisConfig as BidirectionalAnalysisConfig,
    OracleAnalysisPoint,
    OracleAnalysisReport,
)


class BidirectionalAnalysisModel(_OracleAnalysisAdapter):
    def __init__(self, config: BidirectionalAnalysisConfig | None = None):
        super().__init__(config or BidirectionalAnalysisConfig())


__all__ = [
    "BidirectionalAnalysisConfig",
    "BidirectionalAnalysisModel",
    "OracleAnalysisPoint",
    "OracleAnalysisReport",
]
