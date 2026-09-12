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
    assert limits.movetime_ms == 5_000


def test_sliding_pieces_keep_their_move_geometry() -> None:
    bishop = Board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
    rook = Board("4k3/8/8/8/8/8/8/4KR3 w - - 0 1")
    assert all(move.uci() != "f1g1" for move in bishop.legal_moves())
    assert all(move.uci() != "f1g2" for move in rook.legal_moves())


def test_native_board_survives_a_long_legal_sequence() -> None:
    board = Board()
    for _ in range(120):
        moves = board.legal_moves()
        if not moves:
            break
        board.push(moves[0])
    assert board.squares.count("K") == 1
    assert board.squares.count("k") == 1
