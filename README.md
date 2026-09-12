# PaperZero

PaperZero is a lightweight chess engine designed to grow toward a Stockfish-style
engine through small, testable stages.

The first engine slice includes:

- a dependency-free native 0x88 board and legal move generator
- centipawn material and mobility evaluation
- alpha-beta negamax search
- quiescence search and move ordering
- iterative deepening and a transposition table
- a UCI protocol entry point for chess GUIs

## Resource Budget

PaperZero is intentionally small and single-threaded:

- no runtime dependencies
- about 20 KB of engine source in the current version
- a bounded 50,000-entry transposition table
- a hard maximum search time of 2 seconds
- one CPU core by design

## Run

Install the project and its dependency:

```powershell
python -m pip install -e .
```

Run the UCI engine:

```powershell
python -m paperzero_engine.uci
```

Run tests:

```powershell
python -m pytest
```

The long-term roadmap is to replace the 0x88 board with an even faster compact
bitboard representation, then add stronger evaluation, principal variation
search, and pruning without violating the resource budget.
