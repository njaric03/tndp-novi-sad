# Ceo ispis repoa je na srpskom sa dijakritikom, a Windows konzola je
# podrazumevano cp1252 i pukne na prvom "č" — i to tek usred dugog
# eksperimenta, pošto se do tada ništa nije štampalo. Prebacivanje stdout-a
# na utf-8 stoji ovde da bi važilo za svaku skriptu koja išta uveze iz tndp,
# umesto da se ponavlja u svakoj posebno.
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
