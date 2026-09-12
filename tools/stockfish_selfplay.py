"""Generate sharded Stockfish self-play PGNs without holding the dataset in RAM.

Example:
    python tools/stockfish_selfplay.py --stockfish stockfish.exe \
        --games 1000 --workers 8 --movetime-ms 100 \
        --output-dir pgn/stockfish-selfplay

Each worker writes its own compressed PGN shard, so interrupted runs can be
resumed without corrupting one shared archive.
"""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import chess
import chess.pgn
import zstandard as zstd

try:
    from tools.match import UCIProcess, resolve_stockfish
except ModuleNotFoundError:
    from match import UCIProcess, resolve_stockfish



def play_game(stockfish: Path, movetime_ms: int, max_fullmoves: int, root: Path) -> chess.pgn.Game:
    engine = UCIProcess([str(stockfish)], root)
    try:
        engine.send("setoption name Threads value 1")
        engine.send("setoption name Hash value 64")
        engine.send("isready")
        engine._read_until("readyok")
        board = chess.Board()
        moves: list[str] = []
        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_fullmoves:
            move = chess.Move.from_uci(engine.best_move(moves, movetime_ms))
            if move not in board.legal_moves:
                raise RuntimeError(f"Stockfish returned illegal move: {move.uci()}")
            board.push(move)
            moves.append(move.uci())
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "Stockfish 19"
        game.headers["Black"] = "Stockfish 19"
        game.headers["Result"] = board.result(claim_draw=True)
        return game
    finally:
        engine.close()



def write_shard(
    worker: int,
    games: int,
    stockfish: Path,
    output_dir: Path,
    movetime_ms: int,
    max_fullmoves: int,
    root: Path,
) -> Path:
    output = output_dir / f"stockfish-selfplay-{worker:04d}.pgn.zst"
    compressor = zstd.ZstdCompressor(level=3)
    with output.open("wb") as raw, compressor.stream_writer(raw) as compressed:
        for game_number in range(games):
            game = play_game(stockfish, movetime_ms, max_fullmoves, root)
            text = str(game) + "\n\n"
            compressed.write(text.encode("utf-8"))
            if (game_number + 1) % 10 == 0:
                print(f"worker {worker}: {game_number + 1}/{games}", flush=True)
    return output



def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--max-fullmoves", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=Path("pgn/stockfish-selfplay"))
    parser.add_argument("--reserve-free-gb", type=float, default=20.0)
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1:
        raise SystemExit("--games and --workers must be positive")
    if not 1 <= args.movetime_ms <= 5_000:
        raise SystemExit("--movetime-ms must be between 1 and 5000")
    root = Path(__file__).resolve().parents[1]
    stockfish = resolve_stockfish(args.stockfish)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.output_dir).free / 1_000_000_000
    if free_gb <= args.reserve_free_gb:
        raise SystemExit(f"Only {free_gb:.1f} GB free; refusing to start with {args.reserve_free_gb:.1f} GB reserve")
    workers = min(args.workers, args.games)
    counts = [args.games // workers] * workers
    for index in range(args.games % workers):
        counts[index] += 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                write_shard,
                index,
                count,
                stockfish,
                args.output_dir,
                min(args.movetime_ms, 5_000),
                args.max_fullmoves,
                root,
            )
            for index, count in enumerate(counts)
        ]
        outputs = [future.result() for future in futures]
    total_bytes = sum(path.stat().st_size for path in outputs)
    print(f"generated {args.games} games in {len(outputs)} shards")
    print(f"compressed size: {total_bytes / 1_000_000_000:.3f} GB")
    print(f"output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
