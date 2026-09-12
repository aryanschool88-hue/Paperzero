"""Compact dependency-free chess board using a 0x88 mailbox."""

from dataclasses import dataclass

FILES = "abcdefgh"
PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
KNIGHT_STEPS = (-33, -31, -18, -14, 14, 18, 31, 33)
KING_STEPS = (-17, -16, -15, -1, 1, 15, 16, 17)
SLIDERS = (("B", (-17, -15, 15, 17)), ("R", (-16, -1, 1, 16)), ("Q", (-17, -16, -15, -1, 1, 15, 16, 17)))


def square_name(square: int) -> str:
    return FILES[square & 7] + str((square >> 4) + 1)


def parse_square(name: str) -> int:
    return (int(name[1]) - 1) * 16 + FILES.index(name[0])


@dataclass(frozen=True, slots=True)
class Move:
    source: int
    target: int
    promotion: str = ""
    en_passant: bool = False
    castle: bool = False

    def uci(self) -> str:
        return square_name(self.source) + square_name(self.target) + self.promotion.lower()


class Board:
    def __init__(self, fen: str | None = None) -> None:
        self.squares = ["."] * 128
        self.turn = "w"
        self.castling = "KQkq"
        self.en_passant = -1
        self.halfmove = 0
        self.fullmove = 1
        self._history = []
        self.set_fen(fen or "startpos")

    def reset(self) -> None:
        self.set_fen("startpos")

    def set_fen(self, fen: str) -> None:
        if fen == "startpos":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fields = fen.split()
        self.squares = ["."] * 128
        for rank, row in enumerate(fields[0].split("/")):
            file = 0
            for char in row:
                if char.isdigit():
                    file += int(char)
                else:
                    self.squares[(7 - rank) * 16 + file] = char
                    file += 1
        self.turn = fields[1]
        self.castling = "" if fields[2] == "-" else fields[2]
        self.en_passant = -1 if fields[3] == "-" else parse_square(fields[3])
        self.halfmove = int(fields[4])
        self.fullmove = int(fields[5])
        self._history.clear()

    def fen_key(self) -> tuple:
        return (tuple(self.squares), self.turn, self.castling, self.en_passant)

    def piece_at(self, square: int) -> str:
        return self.squares[square]

    def push_uci(self, text: str) -> None:
        move = next(move for move in self.legal_moves() if move.uci() == text)
        self.push(move)

    def push(self, move: Move) -> None:
        state = (self.squares[:], self.turn, self.castling, self.en_passant, self.halfmove, self.fullmove)
        self._history.append(state)
        piece = self.squares[move.source]
        capture = self.squares[move.target] != "."
        self.squares[move.source] = "."
        self.squares[move.target] = move.promotion.upper() if piece.isupper() and move.promotion else move.promotion if piece.islower() and move.promotion else piece
        if move.en_passant:
            self.squares[move.target + (-16 if self.turn == "w" else 16)] = "."
        if move.castle:
            rook_from, rook_to = ((7, 5), (0, 3)) if move.target > move.source else ((0, 3), (7, 5))
            if move.source // 16 == 0:
                rook_from, rook_to = (rook_from[0], rook_to[0])
            else:
                rook_from, rook_to = (rook_from[1], rook_to[1])
            self.squares[rook_to] = self.squares[rook_from]
            self.squares[rook_from] = "."
        self._update_castling(piece, move.source, move.target)
        self.en_passant = move.source + (16 if self.turn == "w" else -16) if piece.upper() == "P" and abs(move.target - move.source) == 32 else -1
        self.halfmove = 0 if piece.upper() == "P" or capture else self.halfmove + 1
        if self.turn == "b":
            self.fullmove += 1
        self.turn = "b" if self.turn == "w" else "w"

    def pop(self) -> None:
        self.squares, self.turn, self.castling, self.en_passant, self.halfmove, self.fullmove = self._history.pop()

    def _update_castling(self, piece: str, source: int, target: int) -> None:
        rights = set(self.castling)
        for char in ("K", "Q") if piece == "K" else ("k", "q") if piece == "k" else ():
            rights.discard(char)
        for square, chars in ((0, "Q"), (7, "K"), (112, "q"), (119, "k")):
            if source == square or target == square:
                rights.discard(chars)
        self.castling = "".join(char for char in "KQkq" if char in rights)

    def legal_moves(self) -> list[Move]:
        result = []
        for move in self._pseudo_moves():
            self.push(move)
            legal = not self.in_check("b" if self.turn == "w" else "w")
            self.pop()
            if legal:
                result.append(move)
        return result

    def captures(self) -> list[Move]:
        return [move for move in self.legal_moves() if self.squares[move.target] != "." or move.en_passant]

    def in_check(self, color: str) -> bool:
        king = "K" if color == "w" else "k"
        square = self.squares.index(king)
        enemy = "b" if color == "w" else "w"
        for move in self._pseudo_moves(color=enemy, attacks_only=True):
            if move.target == square:
                return True
        return False

    def checkmate(self) -> bool:
        return self.in_check(self.turn) and not self.legal_moves()

    def stalemate(self) -> bool:
        return not self.in_check(self.turn) and not self.legal_moves()

    def _pseudo_moves(self, color: str | None = None, attacks_only: bool = False) -> list[Move]:
        color = color or self.turn
        result = []
        for source, piece in enumerate(self.squares):
            if source & 8 or piece == "." or (piece.isupper()) != (color == "w"):
                continue
            kind = piece.upper()
            if kind == "P":
                direction = 16 if color == "w" else -16
                start = 1 if color == "w" else 6
                for target in (source + direction, source + 2 * direction):
                    if not target & 8 and self.squares[target] == "." and not attacks_only:
                        if target // 16 in (0, 7):
                            result.extend(Move(source, target, p) for p in "qrbn")
                        else:
                            result.append(Move(source, target))
                        if source // 16 != start:
                            break
                    else:
                        break
                for offset in (direction - 1, direction + 1):
                    target = source + offset
                    if target & 8:
                        continue
                    target_piece = self.squares[target]
                    if (target_piece != "." and target_piece.isupper() != (color == "w")) or target == self.en_passant:
                        promotion = "q" if target // 16 in (0, 7) else ""
                        result.append(Move(source, target, promotion, target == self.en_passant))
            elif kind == "N":
                result.extend(self._step_moves(source, KNIGHT_STEPS, color, attacks_only))
            elif kind == "K":
                result.extend(self._step_moves(source, KING_STEPS, color, attacks_only))
                if not attacks_only:
                    result.extend(self._castle_moves(color))
            else:
                for slider, offsets in SLIDERS:
                    if kind != "Q" and kind != slider:
                        continue
                    for offset in offsets:
                        target = source + offset
                        while 0 <= target < 128 and not target & 8:
                            target_piece = self.squares[target]
                            if target_piece == ".":
                                result.append(Move(source, target))
                            else:
                                if target_piece.isupper() != (color == "w"):
                                    result.append(Move(source, target))
                                break
                            if kind == "B" and offset not in (-17, -15, 15, 17):
                                break
                            if kind == "R" and offset not in (-16, -1, 1, 16):
                                break
                            target += offset
        return result

    def _step_moves(self, source: int, offsets: tuple, color: str, attacks_only: bool) -> list[Move]:
        result = []
        for offset in offsets:
            target = source + offset
            if not 0 <= target < 128 or target & 8:
                continue
            piece = self.squares[target]
            if piece == "." or piece.isupper() != (color == "w"):
                result.append(Move(source, target))
        return result

    def _castle_moves(self, color: str) -> list[Move]:
        result = []
        rank = 0 if color == "w" else 7
        king = rank * 16 + 4
        for right, rook_file, path in (("K" if color == "w" else "k", 7, (5, 6)), ("Q" if color == "w" else "q", 0, (1, 2, 3))):
            if right not in self.castling or any(self.squares[rank * 16 + file] != "." for file in path):
                continue
            if self.in_check(color) or any(self._attacked(rank * 16 + file, "b" if color == "w" else "w") for file in path[-2:]):
                continue
            result.append(Move(king, rank * 16 + (6 if rook_file else 2), castle=True))
        return result

    def _attacked(self, square: int, by_color: str) -> bool:
        return any(move.target == square for move in self._pseudo_moves(by_color, True))
