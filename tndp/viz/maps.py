import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tndp.viz import style
import numpy as np

from tndp.core.assignment import assign

LINE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
               "#a65628", "#f781bf", "#999999"]


# nacrtaj mrezu linija na gradu: sive ulice u pozadini, cvorovi skalirani traznjom
def draw_network(ax, city, network, title=""):
    for i, j in city.street_edges:
        ax.plot(city.coords[[i, j], 0], city.coords[[i, j], 1],
                color="0.85", lw=0.8, zorder=1)
    w = city.demand.sum(0) + city.demand.sum(1)
    ax.scatter(city.coords[:, 0], city.coords[:, 1],
               s=6 + 60 * w / w.max(), color="0.4", zorder=2)
    for k, route in enumerate(network.routes):
        pts = city.coords[route]
        off = (k - len(network.routes) / 2) * 0.03  # razmak paralelnih linija
        ax.plot(pts[:, 0] + off, pts[:, 1] + off, color=LINE_COLORS[k % 8],
                lw=2.2, zorder=3, label=f"linija {k + 1}", solid_capstyle="round")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


# uporedni prikaz vise mreza na istom gradu (npr
def compare_networks(city, named_networks, out_path, alpha=0.5):
    style.apply_style()
    fig, axes = plt.subplots(1, len(named_networks),
                             figsize=(6 * len(named_networks), 5.5))
    if len(named_networks) == 1:
        axes = [axes]
    for ax, (name, net) in zip(axes, named_networks.items()):
        res = assign(city, net)
        sub = (f"C_p_all {res.C_p_all:.1f} min | C_o {res.C_o:.0f} min "
               f"| d_un {res.d['d_un']:.2f}")
        draw_network(ax, city, net, title=f"{name}\n{sub}")
    fig.suptitle(f"{city.name} (n={city.n}), R={len(next(iter(named_networks.values())).routes)}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
