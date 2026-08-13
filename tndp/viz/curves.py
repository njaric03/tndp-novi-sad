# krive treninga iz runs/<ime>/log.csv pokretanje: python -m tndp.viz.curves runs/smoke

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tndp.viz import style
import numpy as np


def main():
    run = Path(sys.argv[1])
    rows = np.genfromtxt(run / "log.csv", delimiter=",", names=True)

    style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(rows["iter"], rows["reward"], lw=0.8, label="trening")
    has_val = ~np.isnan(rows["val_reward"])
    axes[0].plot(rows["iter"][has_val], rows["val_reward"][has_val],
                 "o-", ms=3, label="validacija (greedy dekod)")
    axes[0].set_title("nagrada")
    axes[0].legend()
    axes[1].plot(rows["iter"], rows["d_un"], lw=0.8)
    axes[1].set_title("d_un (nepokriven demand)")
    axes[2].plot(rows["iter"], rows["entropy"], lw=0.8)
    axes[2].set_title("entropija politike")
    for ax in axes:
        ax.set_xlabel("iteracija")

    fig.tight_layout()
    # runs/ je van gita, pa slika ide i u results/ odakle se predaje
    results = Path(__file__).parent.parent.parent / "results"
    results.mkdir(exist_ok=True)
    out = [run / "curves.png", results / f"curves-{run.name}.png"]
    for p in out:
        fig.savefig(p, dpi=120)
    print("snimljeno u " + " i ".join(str(p) for p in out))


if __name__ == "__main__":
    main()
