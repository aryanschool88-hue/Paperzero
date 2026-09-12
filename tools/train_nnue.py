"""Train PaperZero's optional NNUE-style evaluator from games and Stockfish.

Example:
    python tools/train_nnue.py --pgn games.pgn --stockfish stockfish.exe \
        --output paperzero.nnue.npz --epochs 10
"""

from argparse import ArgumentParser
from contextlib import contextmanager
import io
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess.pgn
import numpy as np
import zstandard as zstd

from paperzero_engine.nnue import NnueModel, encode
try:
    from tools.match import resolve_stockfish
except ModuleNotFoundError:
    from match import resolve_stockfish


def stockfish_score(engine: subprocess.Popen[str], fen: str, depth: int) -> int:
    assert engine.stdin is not None and engine.stdout is not None
    engine.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
    engine.stdin.flush()
    score = 0
    while True:
        line = engine.stdout.readline()
        if not line:
            raise RuntimeError("Stockfish exited while labeling a position")
        if line.startswith("info") and "score" in line:
            fields = line.split()
            score_index = fields.index("score")
            score_type = fields[score_index + 1]
            score_value = int(fields[score_index + 2])
            score = score_value * 100 if score_type == "mate" else score_value
            if score_type == "mate":
                score = 100_000 if score_value > 0 else -100_000
        if line.startswith("bestmove"):
            return score


@contextmanager
def open_pgn(path: Path):
    """Open plain PGN or stream a .pgn.zst archive without extracting it."""
    if path.suffix == ".zst":
        compressed = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        try:
            yield text
        finally:
            text.close()
            compressed.close()
    else:
        with path.open(encoding="utf-8") as text:
            yield text


def collect(
    pgn_paths: list[Path], stockfish_path: Path, limit: int | None, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    engine = subprocess.Popen(
        [str(stockfish_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        assert engine.stdin is not None
        engine.stdin.write("uci\nisready\n")
        engine.stdin.flush()
        while engine.stdout and not engine.stdout.readline().startswith("readyok"):
            pass
        for pgn_path in pgn_paths:
            if limit is not None and len(targets) >= limit:
                break
            with open_pgn(pgn_path) as handle:
                while limit is None or len(targets) < limit:
                    game = chess.pgn.read_game(handle)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        if limit is not None and len(targets) >= limit:
                            break
                        fen = board.fen()
                        score = stockfish_score(engine, fen, depth)
                        features.append(encode_from_fen(fen))
                        targets.append(score)
                        board.push(move)
    finally:
        if engine.stdin:
            engine.stdin.write("quit\n")
            engine.stdin.flush()
        engine.wait(timeout=5)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def encode_from_fen(fen: str) -> np.ndarray:
    from paperzero_engine.board import Board
    return encode(Board(fen))


def expand_pgn_paths(paths: list[Path]) -> list[Path]:
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pgn")))
            expanded.extend(sorted(path.glob("*.pgn.zst")))
        else:
            expanded.append(path)
    return expanded


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paperzero.nnue.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()
    args.stockfish = resolve_stockfish(args.stockfish)
    pgn_paths = expand_pgn_paths(args.pgn)
    if not pgn_paths:
        raise SystemExit("No PGN or PGN.ZST files found")
    if not 1 <= args.depth <= 20:
        raise SystemExit("--depth must be between 1 and 20")
    features, targets = collect(pgn_paths, args.stockfish, args.limit, args.depth)
    if not len(targets):
        raise SystemExit("No positions found in the PGN file")
    model = NnueModel()
    losses = model.fit(features, targets, epochs=args.epochs)
    model.save(args.output)
    print(f"trained {len(targets)} positions; final MSE={losses[-1]:.1f}; saved {args.output}")


if __name__ == "__main__":
    main()
