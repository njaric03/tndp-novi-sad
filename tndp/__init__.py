# Ceo ispis repoa je na srpskom sa dijakritikom, a Windows konzola je podrazumevano cp1252 i pukne na prvom "c"
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# koren repoa i dve putanje koje je do sad svaka skripta racunala za sebe
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCHMARKS = ROOT / "data" / "benchmarks"
