"""Static evaluation for the first PaperZero searcher."""

from .board import Board

PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
CENTER_SQUARES = (51, 52, 67, 68)


def evaluate(board: Board) -> int:
    """Return a score in centipawns from the side-to-move perspective."""
    if board.checkmate():
        return -100000
    if board.stalemate():
        return 0

    score = 0
    for piece, value in PIECE_VALUES.items():
        score += board.squares.count(piece) * value
        score -= board.squares.count(piece.lower()) * value

    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece != ".":
            score += 12 if piece.isupper() else -12

    mobility = len(board.legal_moves())
    board.turn = "b" if board.turn == "w" else "w"
    opponent_mobility = len(board.legal_moves())
    board.turn = "b" if board.turn == "w" else "w"
    score += 2 * (mobility - opponent_mobility)

    return score if board.turn == "w" else -score
