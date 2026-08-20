# zajednicki izgled svih slika: ista metoda ima istu boju na svakoj slici

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# jedna boja po metodi kroz ceo rad
METHOD_COLORS = {
    "random": "#999999",        # siva: donja granica, ne takmac
    "greedy": "#ff7f00",        # narandzasta: konstruktivna heuristika
    "hill climbing": "#4daf4a",  # zelena: lokalna pretraga
    "RL greedy": "#377eb8",     # plava: naucena politika, jedan prolaz
    "RL sampling": "#e41a1c",   # crvena: naucena politika, pretraga
    "MCTS": "#984ea3",          # ljubicasta
    "hibrid": "#a65628",        # braon: politika + lokalna pretraga
    "literatura": "#000000",
}
FALLBACK = "#666666"

# Rad se slaze Latin Modernom, pa figure moraju biti serifne, inace izgledaju
# nalepljeno iz drugog dokumenta. Latin Modern se ne sme pretpostaviti da je
# instaliran; STIXGeneral i DejaVu Serif dolaze uz matplotlib, pa se figure
# pregenerisu isto na svakoj masini. STIX ide prvi jer uz njega ide i isti
# matematicki set, pa $\alpha$ i $C_p$ na slici ne odskacu od teksta oko sebe.
SERIF = ["STIXGeneral", "DejaVu Serif", "Times New Roman"]

# Naziv metode kakav stoji u tekstu rada. Interni kljucevi su engleski jer se
# tako zovu funkcije, ali slika i tekst moraju da koriste isti naziv, inace
# citalac ne zna da su "greedy" i "konstruktivna heuristika" ista stvar.
# Duzi kljucevi prvi: "RL greedy" se poklapa i sa "greedy" prefiksom.
LABELS = [
    ("RL sampling", "politika, uzorkovanje"),
    ("RL greedy", "politika, najverovatniji potez"),
    ("RL", "politika"),
    ("MCTS", "politika, pretraga stabla"),
    ("hill climbing", "lokalna pretraga"),
    ("greedy", "konstruktivna heuristika"),
    ("random", "nasumična pretraga"),
    ("hibrid", "politika kao start"),
    ("GSP", "GSP, postojeća mreža"),
]


def color_for(name):
    for key, c in METHOD_COLORS.items():
        if name.lower().startswith(key.lower()):
            return c
    return FALLBACK


# "RL sampling 32" -> "politika, uzorkovanje 32"; nepoznato ime prolazi netaknuto
def label(name):
    for key, serbian in LABELS:
        if name.lower().startswith(key.lower()):
            rest = name[len(key):].strip()
            # "dekod", "search", "climbing" su vec u prevodu, ne lepe se nazad
            if rest and not rest[0].isdigit():
                rest = ""
            return f"{serbian} {rest}".strip()
    return name


# samo PNG; LaTeX ga prima direktno. vektor po potrebi: formats=("png", "pdf")
def save(fig, out_path, formats=("png",)):
    # nastavak se skida sa IMENA fajla, ne sa cele putanje: direktorijum sme da
    # sadrzi tacku (npr. skriveni direktorijum alata), pa bi rsplit pojeo pola putanje
    out_path = Path(out_path)
    out_path = out_path.with_name(out_path.name.rsplit(".", 1)[0])
    written = []
    for ext in formats:
        p = f"{out_path}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


# Zajednicki izgled svih figura. Poziva se na pocetku svake skripte koja crta,
# da se slike u radu ne razlikuju po fontu, mrezi i debljini linija.
# Decimalni zarez na osama. Tekst rada pise 0,59 a matplotlib podrazumevano
# 0.59, pa se u istoj recenici pojave oba oblika. locale se ne dira: na Windowsu
# srpski locale nije zagarantovan, a i menjao bi ceo proces.
def _zarez(v, _pos=None):
    s = f"{v:g}"
    return s.replace(".", ",") if "e" not in s else s


# za ose koje se ne dobijaju kao Axes (colorbar i slicno)
def zarez_formatter():
    return FuncFormatter(_zarez)


# primeni zarez na obe ose datih osa; poziva se posle podesavanja granica
def decimal_comma(*axes):
    for ax in axes:
        ax.xaxis.set_major_formatter(zarez_formatter())
        ax.yaxis.set_major_formatter(zarez_formatter())


# isto, ali za tekst koji se sam ispisuje na sliku (anotacije, podnaslovi)
def broj(v, decimala=2):
    return f"{v:.{decimala}f}".replace(".", ",")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Figure se u radu skaliraju na sirinu stranice, pa im tekst treba da
        # bude krupniji nego sto izgleda potrebno na ekranu: posle skaliranja
        # od oko 0,75 ovo daje ~8 pt, sto je jos citljivo u stampi.
        "font.size": 11.0,
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "legend.fontsize": 10.0,
        "legend.frameon": False,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "savefig.facecolor": "white",
    })
