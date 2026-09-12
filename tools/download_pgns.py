"""Stream public Lichess broadcast PGN archives into the local corpus.

Examples:
    python tools/download_pgns.py --months 2026-08 2026-09
    python tools/download_pgns.py --url https://example.org/game.pgn.zst
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://database.lichess.org/broadcast"
CHUNK_SIZE = 1024 * 1024


def download(url: str, destination: Path, max_bytes: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "PaperZero PGN collector/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if existing and status != 206:
                existing = 0
                partial.unlink(missing_ok=True)
            mode = "ab" if existing else "wb"
            total = existing
            with partial.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(f"{url} exceeds the {max_bytes / 1e9:.1f} GB limit")
                    output.write(chunk)
                    output.flush()
        partial.replace(destination)
        print(f"downloaded {destination} ({destination.stat().st_size / 1e6:.1f} MB)", flush=True)
        return True
    except (HTTPError, URLError) as error:
        print(f"skipped {url}: {error}", file=sys.stderr)
        return False


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", help="Months such as 2026-08")
    parser.add_argument("--url", action="append", default=[], help="Additional .pgn or .pgn.zst URL")
    parser.add_argument("--output-dir", type=Path, default=Path("pgn/online"))
    parser.add_argument("--max-gb", type=float, default=10.0)
    args = parser.parse_args()
    if not args.months and not args.url:
        raise SystemExit("Provide --months and/or --url")
    urls = list(args.url)
    for month in args.months or []:
        name = f"lichess_db_broadcast_{month}.pgn.zst"
        urls.append(f"{BASE_URL}/{name}")
    max_bytes = int(args.max_gb * 1_000_000_000)
    for url in urls:
        destination = args.output_dir / url.rsplit("/", 1)[-1]
        if destination.exists():
            print(f"already exists {destination}")
            continue
        download(url, destination, max_bytes)


if __name__ == "__main__":
    main()
