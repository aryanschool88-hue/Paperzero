from paperzero_engine.board import Board
from paperzero_engine.search import SearchLimits, Searcher


def test_search_returns_legal_move() -> None:
    board = Board()
    move = Searcher().best_move(board, SearchLimits(depth=2))
    assert move in board.legal_moves()


def test_search_finds_mate_in_one() -> None:
    board = Board("6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1")
    move = Searcher().best_move(board, SearchLimits(depth=2))
    assert move is not None
    board.push(move)
    assert board.checkmate()


def test_uci_position_and_search() -> None:
    board = Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    move = Searcher().best_move(board, SearchLimits(depth=1))
    assert move in board.legal_moves()


def test_resource_limits_are_bounded() -> None:
    limits = SearchLimits()
    assert limits.movetime_ms == 2_000
