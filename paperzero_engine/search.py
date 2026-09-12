"""Single-core classical search with modern bounded alpha-beta techniques."""

from dataclasses import dataclass
import time

from .board import Board, Move, PIECE_VALUES
from .evaluation import evaluate

INF = 1_000_000
MATE_SCORE = 100_000
MAX_SEARCH_MS = 5_000
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


QUIET = 0
LOWER = 1
UPPER = 2


class SearchTimeout(Exception):
    pass


class Searcher:
    def __init__(self, evaluator=None) -> None:
        self.nodes = 0
        self._evaluator = evaluator
        self._deadline: float | None = None
        self._table: dict[tuple, TTEntry] = {}
        self._killers: list[list[Move | None]] = [[None, None] for _ in range(64)]
        self._history: dict[tuple[str, int, int], int] = {}

    def best_move(
        self, board: Board, limits: SearchLimits | None = None
    ) -> Move | None:
        limits = limits or SearchLimits()
        self.nodes = 0
        self._killers = [[None, None] for _ in range(64)]
        self._history.clear()
        budget_ms = min(limits.movetime_ms or MAX_SEARCH_MS, MAX_SEARCH_MS)
        self._deadline = time.monotonic() + budget_ms / 1000
        legal_moves = board.legal_moves()
        if not legal_moves:
            return None

        best_move = legal_moves[0]
        previous_score = 0
        try:
            for depth in range(1, max(1, limits.depth) + 1):
                window = 50 if depth > 3 else INF
                alpha = previous_score - window
                beta = previous_score + window
                score, move = self._root(board, depth, alpha, beta)
                while score <= alpha or score >= beta:
                    score, move = self._root(board, depth, -INF, INF)
                if move is not None:
                    best_move = move
                previous_score = score
                if abs(score) >= MATE_SCORE:
                    break
        except SearchTimeout:
            pass
        finally:
            self._deadline = None
        return best_move

    def _root(self, board: Board, depth: int, alpha: int, beta: int) -> tuple[int, Move | None]:
        best_score = -INF
        best_move = None
        for move_number, move in enumerate(self._ordered_moves(board, ply=0)):
            self._check_time()
            board.push(move)
            if move_number == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_score, best_move

    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int, allow_null: bool = True) -> int:
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
        tt_alpha = alpha
        if cached and cached.depth >= depth:
            if cached.flag == "exact":
                return cached.score
            if cached.flag == "lower":
                alpha = max(alpha, cached.score)
            elif cached.flag == "upper":
                beta = min(beta, cached.score)
            if alpha >= beta:
                return cached.score

        in_check = board.in_check(board.turn)
        if allow_null and depth >= 3 and not in_check and self._has_non_pawn_material(board):
            saved_turn, saved_ep = board.turn, board.en_passant
            board.turn = "b" if saved_turn == "w" else "w"
            board.en_passant = -1
            null_score = -self._negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            board.turn, board.en_passant = saved_turn, saved_ep
            if null_score >= beta:
                return null_score

        best_score = -INF
        best_move = None
        moves = self._ordered_moves(board, cached.move if cached else None, ply=ply)
        for move_number, move in enumerate(moves):
            is_capture = self._is_capture(board, move)
            board.push(move)
            gives_check = board.in_check(board.turn)
            reduction = 1 if depth >= 3 and move_number >= 4 and not gives_check and not move.promotion and not is_capture else 0
            if move_number == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self._negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                if reduction and score > alpha:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                if not is_capture:
                    self._record_cutoff(move, ply, depth, board.turn)
                break

        flag = "exact"
        if best_score <= tt_alpha:
            flag = "upper"
        elif best_score >= beta:
            flag = "lower"
        self._table[key] = TTEntry(depth, best_score, flag, best_move)
        if len(self._table) > MAX_TT_ENTRIES:
            self._table.pop(next(iter(self._table)))
        return best_score

    def _quiescence(self, board: Board, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.nodes & 63 == 0:
            self._check_time()
        if board.checkmate():
            return -MATE_SCORE

        stand_pat = self._evaluate(board)
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
        ply: int = 0,
    ) -> list[Move]:
        candidates = list(moves if moves is not None else board.legal_moves())

        def score(move: Move) -> int:
            value = 0
            if preferred is not None and move == preferred:
                value += 1_000_000
            if self._is_capture(board, move):
                value += 10_000 + self._captured_value(board, move)
            if move.promotion:
                value += 8_000 + PIECE_VALUES[move.promotion.upper()]
            if ply < len(self._killers):
                if move == self._killers[ply][0]:
                    value += 7_000
                elif move == self._killers[ply][1]:
                    value += 6_000
            value += self._history.get((board.turn, move.source, move.target), 0)
            return value

        return sorted(candidates, key=score, reverse=True)

    @staticmethod
    def _captured_value(board: Board, move: Move) -> int:
        piece = board.piece_at(move.target)
        if piece == "." and move.en_passant:
            return 100
        return PIECE_VALUES.get(piece.upper(), 0) if piece != "." else 0

    def _evaluate(self, board: Board) -> int:
        return self._evaluator.evaluate(board) if self._evaluator is not None else evaluate(board)

    @staticmethod
    def _is_capture(board: Board, move: Move) -> bool:
        return board.squares[move.target] != "." or move.en_passant

    def _record_cutoff(self, move: Move, ply: int, depth: int, color: str) -> None:
        if ply < len(self._killers) and move not in self._killers[ply]:
            self._killers[ply][1] = self._killers[ply][0]
            self._killers[ply][0] = move
        key = (color, move.source, move.target)
        self._history[key] = min(32_000, self._history.get(key, 0) + depth * depth)

    @staticmethod
    def _has_non_pawn_material(board: Board) -> bool:
        return any(board.squares.count(piece) for piece in "NBRQnbrq")

    def _check_time(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise SearchTimeout
