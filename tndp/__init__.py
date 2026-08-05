# Ceo ispis repoa je na srpskom sa dijakritikom, a Windows konzola je podrazumevano cp1252 i pukne na prvom "č"
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
