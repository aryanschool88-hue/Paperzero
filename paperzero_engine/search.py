"""Alpha-beta search with iterative deepening and a small transposition table."""

from dataclasses import dataclass
import time

from .board import Board, Move, PIECE_VALUES
from .evaluation import evaluate

INF = 1_000_000
MATE_SCORE = 100_000
MAX_SEARCH_MS = 2_000
MAX_TT_ENTRIES = 50_000


@dataclass(slots=True)
class SearchLimits:
    depth: int = 4
    movetime_ms: int | None = MAX_SEARCH_MS


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    flag: str
    move: Move | None


class SearchTimeout(Exception):
    pass


class Searcher:
    def __init__(self) -> None:
        self.nodes = 0
        self._deadline: float | None = None
        self._table: dict[tuple, TTEntry] = {}

    def best_move(
        self, board: Board, limits: SearchLimits | None = None
    ) -> Move | None:
        limits = limits or SearchLimits()
        self.nodes = 0
        self._deadline = (
            time.monotonic() + min(limits.movetime_ms or MAX_SEARCH_MS, MAX_SEARCH_MS) / 1000
            if limits.movetime_ms is not None or MAX_SEARCH_MS
            else None
        )
        legal_moves = board.legal_moves()
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        try:
            for depth in range(1, max(1, limits.depth) + 1):
                score, move = self._root(board, depth)
                if move is not None:
                    best_move = move
                if abs(score) >= MATE_SCORE:
                    break
        except SearchTimeout:
            pass
        finally:
            self._deadline = None
        return best_move

    def _root(self, board: Board, depth: int) -> tuple[int, Move | None]:
        best_score = -INF
        best_move = None
        alpha = -INF
        for move in self._ordered_moves(board):
            self._check_time()
            board.push(move)
            score = -self._negamax(board, depth - 1, -INF, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_score, best_move

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.nodes & 63 == 0:
            self._check_time()
        if board.checkmate():
            return -MATE_SCORE - depth
        if board.stalemate():
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        key = board.fen_key()
        original_alpha = alpha
        cached = self._table.get(key)
        if cached and cached.depth >= depth:
            if cached.flag == "exact":
                return cached.score
            if cached.flag == "lower":
                alpha = max(alpha, cached.score)
            elif cached.flag == "upper":
                beta = min(beta, cached.score)
            if alpha >= beta:
                return cached.score

        best_score = -INF
        best_move = None
        for move in self._ordered_moves(board, cached.move if cached else None):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        flag = "exact"
        if best_score <= original_alpha:
            flag = "upper"
        elif best_score >= beta:
            flag = "lower"
        self._table[key] = TTEntry(depth, best_score, flag, best_move)
        if len(self._table) > MAX_TT_ENTRIES:
            self._table.clear()
        return best_score

    def _quiescence(self, board: Board, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.nodes & 63 == 0:
            self._check_time()
        if board.checkmate():
            return -MATE_SCORE

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        moves = board.legal_moves() if board.in_check(board.turn) else board.captures()
        for move in self._ordered_moves(board, moves=moves):
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _ordered_moves(
        self,
        board: Board,
        preferred: Move | None = None,
        moves=None,
    ) -> list[Move]:
        candidates = list(moves if moves is not None else board.legal_moves())

        def score(move: Move) -> int:
            value = 0
            if preferred is not None and move == preferred:
                value += 1_000_000
            if board.squares[move.target] != "." or move.en_passant:
                value += 10_000 + self._captured_value(board, move)
            if move.promotion:
                value += 8_000 + PIECE_VALUES[move.promotion.upper()]
            return value

        return sorted(candidates, key=score, reverse=True)

    @staticmethod
    def _captured_value(board: Board, move: Move) -> int:
        piece = board.piece_at(move.target)
        if piece == "." and move.en_passant:
            return 100
        return PIECE_VALUES.get(piece.upper(), 0) if piece != "." else 0

    def _check_time(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise SearchTimeout
