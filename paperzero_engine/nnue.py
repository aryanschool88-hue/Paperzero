"""Optional compact NNUE-style evaluator and trainer.

This module is deliberately separate from the dependency-free classical engine.
It uses a small feature transformer and one hidden layer, not Stockfish's
licensed network format.
"""

from pathlib import Path

import numpy as np

from .board import Board

FEATURES = 769
HIDDEN = 128


def encode(board: Board) -> np.ndarray:
    """Encode a position as piece-square one-hot features plus side to move."""
    result = np.zeros(FEATURES, dtype=np.float32)
    piece_ids = {piece: index for index, piece in enumerate("PNBRQKpnbrqk")}
    for square in range(128):
        if square & 8:
            continue
        piece = board.squares[square]
        if piece != ".":
            result[piece_ids[piece] * 64 + (square // 16) * 8 + (square & 7)] = 1.0
    result[-1] = 1.0 if board.turn == "w" else 0.0
    return result


class NnueModel:
    """Small CPU-friendly neural evaluator trained on centipawn targets."""

    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_weights = (rng.standard_normal((FEATURES, HIDDEN), dtype=np.float32) * 0.02)
        self.hidden_bias = np.zeros(HIDDEN, dtype=np.float32)
        self.output_weights = (rng.standard_normal(HIDDEN, dtype=np.float32) * 0.02)
        self.output_bias = np.float32(0.0)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, features @ self.hidden_weights + self.hidden_bias)
        return hidden @ self.output_weights + self.output_bias

    def evaluate(self, board: Board) -> int:
        score = float(self.predict_features(encode(board)))
        return int(np.clip(score, -100_000, 100_000))

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        target_clip: float = 5_000.0,
        error_clip: float = 1_000.0,
    ) -> list[float]:
        """Train with clipped MSE and return the mean loss per epoch."""
        losses = []
        for _ in range(epochs):
            order = np.random.permutation(len(features))
            total = 0.0
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                x = features[batch]
                y = np.clip(targets[batch].astype(np.float32), -target_clip, target_clip)
                pre_activation = x @ self.hidden_weights + self.hidden_bias
                hidden = np.maximum(0.0, pre_activation)
                prediction = hidden @ self.output_weights + self.output_bias
                error = np.clip(prediction - y, -error_clip, error_clip)
                scale = 2.0 / max(1, len(batch))
                output_gradient = scale * hidden.T @ error
                hidden_gradient = scale * (error[:, None] * self.output_weights) * (pre_activation > 0)
                self.output_weights -= learning_rate * output_gradient
                self.output_bias -= learning_rate * error.sum()
                self.hidden_weights -= learning_rate * (x.T @ hidden_gradient)
                self.hidden_bias -= learning_rate * hidden_gradient.sum(axis=0)
                total += float(np.mean(error * error)) * len(batch)
            losses.append(total / max(1, len(order)))
        return losses

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            hidden_weights=self.hidden_weights,
            hidden_bias=self.hidden_bias,
            output_weights=self.output_weights,
            output_bias=self.output_bias,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NnueModel":
        data = np.load(path)
        model = cls()
        model.hidden_weights = data["hidden_weights"]
        model.hidden_bias = data["hidden_bias"]
        model.output_weights = data["output_weights"]
        model.output_bias = data["output_bias"]
        return model
