"""UCI protocol adapter for running PaperZero in chess GUIs."""

import sys

from .board import Board
from .search import SearchLimits, Searcher


class UCIEngine:
    def __init__(self) -> None:
        self.board = Board()
        self.searcher = Searcher()

    def handle(self, line: str) -> bool:
        tokens = line.strip().split()
        if not tokens:
            return True
        command = tokens[0]
        if command == "uci":
            print("id name PaperZero")
            print("id author PaperZero contributors")
            print("uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            self.board.reset()
            self.searcher = Searcher()
        elif command == "position":
            self._set_position(tokens[1:])
        elif command == "go":
            self._go(tokens[1:])
        elif command == "quit":
            return False
        return True

    def _set_position(self, tokens: list[str]) -> None:
        if not tokens:
            return
        if tokens[0] == "startpos":
            self.board.reset()
            move_start = 1
        elif tokens[0] == "fen":
            fen_end = tokens.index("moves") if "moves" in tokens else len(tokens)
            self.board.set_fen(" ".join(tokens[1:fen_end]))
            move_start = fen_end
        else:
            return
        if move_start < len(tokens) and tokens[move_start] == "moves":
            for san_move in tokens[move_start + 1:]:
                self.board.push_uci(san_move)

    def _go(self, tokens: list[str]) -> None:
        depth = 4
        movetime = None
        if "depth" in tokens:
            depth = int(tokens[tokens.index("depth") + 1])
        if "movetime" in tokens:
            movetime = int(tokens[tokens.index("movetime") + 1])
        move = self.searcher.best_move(
            self.board, SearchLimits(depth=depth, movetime_ms=movetime)
        )
        print(f"bestmove {move.uci() if move else '0000'}", flush=True)


def main() -> None:
    engine = UCIEngine()
    for line in sys.stdin:
        if not engine.handle(line):
            break


if __name__ == "__main__":
    main()
