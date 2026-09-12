"""Launch PaperZero with the final trained NNUE model preloaded."""

from pathlib import Path
import sys

from paperzero_engine.uci import UCIEngine
from paperzero_engine.nnue import NnueModel
from paperzero_engine.search import Searcher


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "paperzero-final.nnue.npz"


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"NNUE model not found: {MODEL}")
    engine = UCIEngine()
    engine.nnue = NnueModel.load(MODEL)
    engine.searcher = Searcher(engine.nnue)
    for line in sys.stdin:
        if not engine.handle(line):
            break


if __name__ == "__main__":
    main()
