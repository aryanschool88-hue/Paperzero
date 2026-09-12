# PaperZero

PaperZero is a lightweight chess engine designed to grow toward a Stockfish-style
engine through small, testable stages.

The first engine slice includes:

- a dependency-free native 0x88 board and legal move generator
- centipawn material and mobility evaluation
- alpha-beta negamax search
- principal variation search with aspiration windows
- quiescence search and capture ordering
- null-move pruning with zugzwang-safe guards
- late-move reductions
- killer moves and history heuristic ordering
- iterative deepening and a transposition table
- a UCI protocol entry point for chess GUIs
- classical evaluation in the default build; NNUE is optional

NNUE is now available as an optional, larger build. The default installation
still has no neural dependency; install the training extras with:

```powershell
python -m pip install -e ".[nnue]"
```

Train from human games labeled by Stockfish. Plain `.pgn` and compressed
`.pgn.zst` archives are supported without extracting the archive:

```powershell
python tools/train_nnue.py --pgn games.pgn --stockfish stockfish.exe \
	--output paperzero.nnue.npz --epochs 10
```

For effectively unlimited data without a 50 GB dataset, train online from
Stockfish. Each position is generated, labeled, used in a minibatch, and then
discarded; only the checkpoint stays on disk:

```powershell
python tools/train_online.py --stockfish stockfish.exe `
	--steps 1000000 --batch-size 256 --label-depth 6 `
	--checkpoint-every 10000 --output paperzero-online.nnue.npz
```

This keeps RAM bounded and can resume by pointing `--output` at an existing
checkpoint. The online path is the recommended way to grow training data on a
machine with limited storage.

Load a trained model in a UCI GUI with:

```text
setoption name NNUEFile value paperzero.nnue.npz
```

For the final trained model, use the included launcher. In the GUI set:

```text
Executable: C:\Users\ChaudharyA\AppData\Local\Python\bin\python.exe
Arguments: C:\Paperzero\PaperZero\run_paperzero_final.py
Working directory: C:\Paperzero\PaperZero
```

The launcher automatically loads `paperzero-final.nnue.npz`, so no separate
NNUE option is required.

Run PaperZero versus Stockfish and save the games as PGN:

```powershell
python tools/match.py --stockfish stockfish.exe --games 10 `
	--movetime-ms 5000 --output matches.pgn
```

The match runner uses one thread per engine, enforces legal moves, alternates
colors, and records every result. Strength should be measured with match
statistics and tactical test suites rather than assumed from one game.

Generate large Stockfish self-play data in compressed shards. This computer
has about 15.6 GB RAM, 10 logical cores, and 121.95 GB free on `C:`, so a
50 GB corpus fits on disk but must be streamed in shards:

```powershell
python tools/stockfish_selfplay.py --stockfish stockfish.exe `
	--games 100000 --workers 8 --movetime-ms 100 `
	--output-dir pgn/stockfish-selfplay
```

The generator never loads the corpus into memory, keeps one Stockfish thread
per worker, and reserves 20 GB of free disk space. The NNUE trainer accepts
multiple `.pgn` or `.pgn.zst` files with repeated `--pgn` arguments; use
`--limit` while testing because its current batch trainer materializes labels
in RAM. A shard directory is also accepted directly:

```powershell
python tools/train_nnue.py --pgn pgn/stockfish-selfplay `
	--stockfish stockfish.exe --limit 1000000
```

## Resource Budget

PaperZero is intentionally small and single-threaded:

- no runtime dependencies or NNUE model files in the default build
- about 20 KB of engine source in the current version
- a bounded 50,000-entry transposition table
- a hard maximum search time of 5 seconds
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

Potential research experiments are kept measurable and optional: ProbCut with
strict tactical guards, multi-cut pruning, singular extensions, history
gravity, and adaptive time management. Each experiment must preserve legal
move correctness, the 5-second cap, and one-core execution before becoming
part of the default search.
