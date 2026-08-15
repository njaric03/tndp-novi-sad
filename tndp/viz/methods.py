import matplotlib.pyplot as plt

from tndp.viz import style
import numpy as np


# poredjenje metoda u tri panela: cilj, pokrivenost putovanja i kompromis putnik/operater
def plot_synth(stats, out_path):
    names = list(stats)
    x = np.arange(len(names))

    style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # boja po metodi iz style.METHOD_COLORS, da ista metoda ima istu boju na svakoj slici
    axes[0].bar(x, [stats[n]["cilj"] for n in names],
                color=[style.color_for(n) for n in names])
    axes[0].set_title("cilj (manje je bolje)")

    axes[1].bar(x - 0.2, [stats[n]["d_0"] for n in names], 0.4,
                label="d_0 direktna", color="tab:blue")
    axes[1].bar(x + 0.2, [stats[n]["d_un"] for n in names], 0.4,
                label="d_un nepokriveno", color="tab:red")
    axes[1].set_title("pokrivenost putovanja")
    axes[1].legend()

    axes[2].bar(x - 0.2, [stats[n]["C_p_all"] for n in names], 0.4,
                label="C_p_all putnik (min)", color="tab:blue")
    axes[2].bar(x + 0.2, [stats[n]["C_o"] / 10 for n in names], 0.4,
                label="C_o operater (min/10)", color="tab:orange")
    axes[2].set_title("kompromis putnik/operater")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
