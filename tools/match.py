"""Run reproducible PaperZero versus Stockfish matches.

Example:
    python tools/match.py --stockfish stockfish.exe --games 10 --movetime-ms 5000
"""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys
import shutil

import chess
import chess.pgn


class UCIProcess:
    def __init__(self, command: list[str], cwd: Path) -> None:
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self._read_until("uciok")
        self.send("isready")
        self._read_until("readyok")

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, marker: str) -> list[str]:
        assert self.process.stdout is not None
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited before {marker}")
            lines.append(line.rstrip())
            if line.startswith(marker):
                return lines

    def best_move(self, moves: list[str], movetime_ms: int) -> str:
        position = "position startpos" + (" moves " + " ".join(moves) if moves else "")
        self.send(position)
        self.send(f"go movetime {movetime_ms}")
        for line in self._read_until("bestmove"):
            if line.startswith("bestmove "):
                return line.split()[1]
        raise RuntimeError("engine did not return a move")

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
            self.process.wait(timeout=10)


def resolve_stockfish(path: Path) -> Path:
    """Resolve a command name across PowerShell aliases and Python PATHs."""
    if path.exists():
        return path.resolve()
    found = shutil.which(str(path))
    if found:
        return Path(found).resolve()
    candidates = [
        Path.home() / "bin" / "stockfish.exe",
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages" / "Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/stockfish/stockfish-windows-arm64-universal.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find Stockfish executable: {path}")

def configure(engine: UCIProcess, nnue: Path | None = None) -> None:
    engine.send("setoption name Threads value 1")
    engine.send("setoption name Hash value 32")
    if nnue:
        engine.send(f"setoption name NNUEFile value {nnue}")
    engine.send("isready")
    engine._read_until("readyok")


def play_game(
    root: Path,
    stockfish: Path,
    movetime_ms: int,
    nnue: Path | None,
    paper_white: bool,
    max_fullmoves: int,
) -> chess.pgn.Game:
    paper = UCIProcess([sys.executable, "-m", "paperzero_engine.uci"], root)
    stock = UCIProcess([str(stockfish)], root)
    try:
        configure(paper, nnue)
        configure(stock)
        board = chess.Board()
        moves: list[str] = []
        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_fullmoves:
            engine = paper if (board.turn == chess.WHITE) == paper_white else stock
            uci_move = engine.best_move(moves, movetime_ms)
            if uci_move == "0000":
                if board.is_checkmate() or board.is_stalemate():
                    break
                raise RuntimeError("engine returned 0000 for a non-terminal position")
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                raise RuntimeError(f"illegal move {uci_move} from {engine.process.args}")
            board.push(move)
            moves.append(uci_move)
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "PaperZero" if paper_white else "Stockfish 19"
        game.headers["Black"] = "Stockfish 19" if paper_white else "PaperZero"
        game.headers["Result"] = board.result(claim_draw=True)
        return game
    finally:
        paper.close()
        stock.close()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--movetime-ms", type=int, default=5_000)
    parser.add_argument("--nnue", type=Path)
    parser.add_argument("--max-fullmoves", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("paperzero-v-stockfish.pgn"))
    args = parser.parse_args()
    stockfish = resolve_stockfish(args.stockfish)
    root = Path(__file__).resolve().parents[1]
    results: list[str] = []
    with args.output.open("w", encoding="utf-8") as output:
        for index in range(args.games):
            game = play_game(
                root,
                stockfish,
                min(args.movetime_ms, 5_000),
                args.nnue,
                index % 2 == 0,
                args.max_fullmoves,
            )
            output.write(str(game) + "\n\n")
            results.append(game.headers["Result"])
            print(f"game {index + 1}/{args.games}: {game.headers['Result']}")
    print(f"results: {', '.join(results)}")
    print(f"PGN saved to {args.output}")


if __name__ == "__main__":
    main()
