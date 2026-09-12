"""Train PaperZero NNUE from positions generated live by Stockfish.

This is online distillation: positions are generated, labeled, trained, and
then discarded. Disk usage stays near the model/checkpoint size.
"""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

from paperzero_engine.nnue import NnueModel, encode
try:
    from tools.match import UCIProcess, resolve_stockfish
    from tools.train_nnue import stockfish_score
except ModuleNotFoundError:
    from match import UCIProcess, resolve_stockfish
    from train_nnue import stockfish_score


def random_opening(board: chess.Board, plies: int) -> list[str]:
    moves: list[str] = []
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = random.choice(legal)
        board.push(move)
        moves.append(move.uci())
    return moves


def generate_position(
    engine: UCIProcess,
    label_engine,
    max_fullmoves: int,
    opening_plies: int,
    label_depth: int,
) -> tuple[np.ndarray, int] | None:
    board = chess.Board()
    moves = random_opening(board, random.randint(0, opening_plies))
    continuation = random.randint(1, min(24, max(1, max_fullmoves * 2)))
    for _ in range(continuation):
        if board.is_game_over(claim_draw=True) or board.fullmove_number > max_fullmoves:
            break
        bestmove = engine.best_move(moves, 20)
        if bestmove == "0000":
            break
        move = chess.Move.from_uci(bestmove)
        if move not in board.legal_moves:
            return None
        board.push(move)
        moves.append(bestmove)
    if board.is_game_over(claim_draw=True):
        return None
    fen = board.fen()
    target = stockfish_score(label_engine, fen, label_depth)
    return encode_from_fen(fen), target


def encode_from_fen(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    from paperzero_engine.board import Board
    return encode(Board(fen))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--label-depth", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--output", type=Path, default=Path("paperzero-online.nnue.npz"))
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--opening-plies", type=int, default=12)
    parser.add_argument("--max-fullmoves", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--steps and --batch-size must be positive")
    if not 1 <= args.label_depth <= 20:
        raise SystemExit("--label-depth must be between 1 and 20")

    random.seed(7)
    np.random.seed(7)
    stockfish = resolve_stockfish(args.stockfish)
    root = Path(__file__).resolve().parents[1]
    generator = UCIProcess([str(stockfish)], root)
    labeler = UCIProcess([str(stockfish)], root)
    model = NnueModel.load(args.output) if args.output.exists() else NnueModel(seed=7)
    features: list[np.ndarray] = []
    targets: list[int] = []
    started = time.monotonic()
    completed = 0
    try:
        generator.send("setoption name Threads value 1")
        labeler.send("setoption name Threads value 1")
        for step in range(args.steps):
            sample = generate_position(
                generator, labeler.process, args.max_fullmoves,
                args.opening_plies, args.label_depth,
            )
            if sample is None:
                continue
            feature, target = sample
            features.append(feature)
            targets.append(target)
            if len(features) >= args.batch_size:
                loss = model.fit(
                    np.asarray(features, dtype=np.float32),
                    np.asarray(targets, dtype=np.float32),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    target_clip=5_000.0,
                    error_clip=1_000.0,
                )[-1]
                completed += len(features)
                features.clear()
                targets.clear()
                if completed % args.checkpoint_every < args.batch_size:
                    model.save(args.output)
                    elapsed = max(0.001, time.monotonic() - started)
                    print(f"positions={completed} loss={loss:.1f} rate={completed / elapsed:.1f}/s", flush=True)
        if features:
            model.fit(
                np.asarray(features), np.asarray(targets),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                target_clip=5_000.0,
                error_clip=1_000.0,
            )
        model.save(args.output)
        print(f"completed {completed + len(features)} online positions; saved {args.output}")
    finally:
        generator.close()
        labeler.close()


if __name__ == "__main__":
    main()
