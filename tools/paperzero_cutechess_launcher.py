import subprocess
import sys

PYTHON = r"C:\Users\ChaudharyA\AppData\Local\Python\bin\python.exe"
ENGINE = r"C:\Paperzero\PaperZero\run_paperzero_final.py"

process = subprocess.Popen(
    [PYTHON, ENGINE],
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
)
raise SystemExit(process.wait())
