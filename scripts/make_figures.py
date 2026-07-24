"""Generate the primary manuscript figures (I18) from sealed analysis artifacts.

Every value plotted here is read from a script-generated, immutable JSON under
``artifacts/analysis/<run>/`` -- the same artifacts the claims ledger cites. No
statistic is computed by hand and none is recomputed from raw traces; this
module only reshapes recorded numbers into figure geometry (AGENTS.md).

Pull the inputs first::

    modal volume get cas-artifacts analysis/<run>/<file>.json artifacts/analysis/<run>/

Then::

    python scripts/make_figures.py            # writes paper/figures/*.pdf

Figures produced:
  fig_atlas.pdf   -- C04 acceptance atlas, category x domain (frozen test)
  fig_dose.pdf    -- C03 dose-response steering vs matched controls
  fig_length.pdf  -- C10 per-length survival lift (first-token-only scoping)
  fig_forest.pdf  -- C10 frozen-test lift under both protocols, with controls
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "artifacts" / "analysis"
OUTDIR = REPO / "paper" / "figures"

# (label, run directory). Order is fixed and used by every figure so a reader
# comparing panels across figures sees the settings in the same places.
SETTINGS = [
    ("Qwen-v1", "sweep-2026-07-11T203836"),
    ("Qwen-v2", "sweep-v2-f8-2026-07-13"),
    ("Llama-v1", "sweep-llama-f8-2026-07-13"),
]

# Categorical slots 1/2/3 of the reference palette (blue / orange / aqua),
# assigned in fixed order and never cycled. That ordering is documented as
# passing the adjacent-pair CVD and normal-vision gates in light mode; these are
# print figures, so only the light steps are used.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_MUTED = "#1a1a19", "#4a4a46", "#8a8a80"
GRID = "#e3e3de"

# Sequential ramp for magnitude (single hue, light -> dark), steps 100..700 of
# the blue ramp. Used only for the atlas heatmap.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("cas_blue", BLUE_RAMP)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load(run: str, name: str) -> dict:
    path = ANALYSIS / run / name
    if not path.exists():
        raise SystemExit(
            f"missing artifact: {path}\n"
            f"pull it with: modal volume get cas-artifacts analysis/{run}/{name} {path}"
        )
    with path.open() as fh:
        return json.load(fh)


def tidy(ax, *, grid_axis="y"):
    """Recessive grid and axes; no chartjunk."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)


def prettify(key: str) -> str:
    return key.replace("_", " ")


# --------------------------------------------------------------------------
# Figure 1 -- C04 acceptance atlas (frozen test)
# --------------------------------------------------------------------------
def fig_atlas() -> None:
    """Acceptance rate by token category x domain, one panel per setting.

    Sequential single-hue encoding: the quantity is a magnitude (acceptance
    rate), so it gets a magnitude ramp, not categorical hues. Cells below the
    artifact's own ``min_n`` are simply absent from the JSON and render blank.
    """
    data = [(label, load(run, "c04_atlas_test.json")) for label, run in SETTINGS]

    # Row order: pooled HIGH categories first, pooled LOW last, everything else
    # between -- so the pre-registered contrast is legible as a block, ordered
    # within each block by the Qwen-v1 marginal rate.
    pools = data[0][1]["contrast_pools"]
    high, low = pools["high"], pools["low"]
    ref = {c["key"]: c["rate"] for c in data[0][1]["category_marginal"]}
    every = sorted({c["key"] for _, d in data for c in d["category_marginal"]})
    middle = sorted([k for k in every if k not in high and k not in low],
                    key=lambda k: -ref.get(k, 0.0))
    rows = list(high) + middle + list(low)

    widths = [len(d["within_domain_category"]) for _, d in data]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.1, 4.5), gridspec_kw={"width_ratios": widths})

    for ax, (label, d) in zip(axes, data):
        domains = sorted(d["within_domain_category"])
        grid = np.full((len(rows), len(domains)), np.nan)
        for j, dom in enumerate(domains):
            cell = {c["key"]: c["rate"] for c in d["within_domain_category"][dom]}
            for i, cat in enumerate(rows):
                if cat in cell:
                    grid[i, j] = cell[cat]

        im = ax.imshow(grid, cmap=SEQ, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=45, ha="right")
        ax.set_title(f"{label}\npooled HIGH {d['contrast_overall']['rate_high']:.2f} / "
                     f"LOW {d['contrast_overall']['rate_low']:.2f}", pad=6)
        ax.set_yticks(range(len(rows)))
        # Pool membership rides in the tick label itself: it cannot collide with
        # the marks, and identity is never carried by position alone.
        labels = [prettify(r) + (" (HIGH)" if r in high else
                                 " (LOW)" if r in low else "") for r in rows]
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Direct labels: the cell value in ink that contrasts with its fill.
        for i in range(len(rows)):
            for j in range(len(domains)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="#ffffff" if v > 0.62 else INK)

        # 2px surface gap between cells.
        ax.set_xticks(np.arange(-0.5, len(domains)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)

    # Block separators around the pre-registered HIGH and LOW pools.
    for ax in axes:
        for y in (len(high) - 0.5, len(rows) - len(low) - 0.5):
            ax.axhline(y, color=INK, linewidth=0.9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("first-token acceptance rate", fontsize=7.5, color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=7)

    out = OUTDIR / "fig_atlas.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 2 -- C03 dose-response steering
# --------------------------------------------------------------------------
def fig_dose() -> None:
    """Realized acceptance vs steering dose, real direction vs matched controls.

    Only two settings appear: interventions were run on Qwen-v1 and Llama-v1
    only, so there is no Qwen-v2 panel to omit or to hide.
    """
    runs = [(label, run) for label, run in SETTINGS
            if (ANALYSIS / run / "intervene.json").exists()]
    data = [(label, load(run, "intervene.json")) for label, run in runs]

    series = [("real", "acceptance direction", C_BLUE, "o", "-"),
              ("random", "norm-matched random", C_ORANGE, "s", "--"),
              ("shuffled", "shuffled direction", C_AQUA, "^", ":")]

    layers = sorted(data[0][1]["layers"], key=int)
    fig, axes = plt.subplots(len(data), len(layers), figsize=(7.1, 3.7),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for r, (label, d) in enumerate(data):
        alphas = d["alphas"]
        for c, layer in enumerate(layers):
            ax = axes[r, c]
            per_dir = d["layers"][layer]["per_dir"]
            for key, _, color, marker, ls in series:
                ax.plot(alphas, per_dir[key]["accept_rate"], color=color,
                        linewidth=1.6, linestyle=ls, marker=marker,
                        markersize=3.6, markeredgecolor="white",
                        markeredgewidth=0.7, zorder=3)
            ax.axvline(0.0, color=INK_MUTED, linewidth=0.6, linestyle="-",
                       zorder=1)
            tidy(ax, grid_axis="both")
            if r == 0:
                ax.set_title(f"layer {layer}", pad=4)
            if c == 0:
                ax.set_ylabel(f"{label}\nacceptance", linespacing=1.4)
            if r == len(data) - 1:
                ax.set_xlabel(r"dose $\alpha$")
            ax.set_xticks(alphas)

    handles = [plt.Line2D([], [], color=color, marker=marker, linestyle=ls,
                          linewidth=1.6, markersize=3.6, label=name)
               for _, name, color, marker, ls in series]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()

    out = OUTDIR / "fig_dose.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 3 -- C10 per-length survival lift
# --------------------------------------------------------------------------
def fig_length() -> None:
    """Delta AUROC vs survival length k, with CI bands and a zero reference.

    The story is the decay to zero, so zero is a labelled reference line rather
    than just an axis crossing.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    markers = ["o", "s", "^"]

    for (label, run), color, marker in zip(SETTINGS, colors, markers):
        d = load(run, "autoresearch_length_dev_domctl.json")
        res = next(r for r in d["results"] if r["spec"]["name"] == "raw_frontier")
        ks = [int(k) for k in sorted(res["per_k"], key=int)]
        mid = [res["per_k"][str(k)]["delta_auroc"] for k in ks]
        lo = [res["per_k"][str(k)]["delta_auroc_ci"]["lo"] for k in ks]
        hi = [res["per_k"][str(k)]["delta_auroc_ci"]["hi"] for k in ks]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.plot(ks, mid, color=color, linewidth=1.6, marker=marker,
                markersize=4, markeredgecolor="white", markeredgewidth=0.7,
                label=label, zorder=3)

    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
    tidy(ax)
    ax.set_xlabel(r"survival length $k$  (accepted run $\geq k$)")
    ax.set_ylabel(r"$\Delta$AUROC over baseline")
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.legend(loc="upper right")

    out = OUTDIR / "fig_length.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 4 -- C10 frozen-test forest plot
# --------------------------------------------------------------------------
def fig_forest() -> None:
    """Frozen-test lift under both protocols, against equal-capacity controls.

    P1 is within-test OOF; P2 is the strict dev->test transfer. Control markers
    are the control's own lift over the same baseline -- the point being that
    they sit at zero while the real probe does not.
    """
    rows = []
    for label, run in SETTINGS:
        for proto, fname in (("P2 dev$\\to$test", "autoresearch_frozen_transfer_domctl.json"),
                             ("P1 within-test", "autoresearch_test_domctl.json")):
            d = load(run, fname)
            res = d["results"][0]
            ctrl = max(res["control_random"]["auroc"], res["control_norm"]["auroc"])
            rows.append({
                "label": f"{label}  {proto}",
                "delta": res["deltas"]["auroc"],
                "lo": res["delta_auroc_ci"]["lo"],
                "hi": res["delta_auroc_ci"]["hi"],
                "ctrl": ctrl - res["base"]["auroc"],
            })

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ys = np.arange(len(rows))[::-1]

    for y, row in zip(ys, rows):
        ax.plot([row["lo"], row["hi"]], [y, y], color=C_BLUE, linewidth=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot([row["delta"]], [y], marker="o", markersize=5, color=C_BLUE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.plot([row["ctrl"]], [y], marker="D", markersize=4.2, color=C_ORANGE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    ax.axvline(0.0, color=INK, linewidth=0.8, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel(r"$\Delta$AUROC over pre-round baseline")
    tidy(ax, grid_axis="x")
    ax.set_ylim(-0.5, len(rows) - 0.3)

    handles = [
        plt.Line2D([], [], color=C_BLUE, marker="o", markersize=5, linewidth=1.6,
                   label="frontier representation (95% CI)"),
        plt.Line2D([], [], color=C_ORANGE, marker="D", markersize=4.2,
                   linestyle="none", label="equal-capacity random control"),
    ]
    # Below the plot: the rows span the full width, so an inset legend collides.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=1)

    out = OUTDIR / "fig_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 5 -- calibration summary
# --------------------------------------------------------------------------
def fig_calibration() -> None:
    """Expected calibration error before and after the global Platt map.

    NOTE: this is a calibration *summary*, not a reliability diagram. The sealed
    artifacts record scalar ECE/Brier only -- no per-bin observed-vs-predicted
    counts and no per-example calibrated predictions. Producing a true
    reliability diagram would require re-fitting the probe and re-scoring the
    frozen test split, which the protocol permits exactly once and which has
    already been spent. See the backlog note for the decision this needs.
    """
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ys = np.arange(len(SETTINGS))[::-1]

    for y, (label, run) in zip(ys, SETTINGS):
        res = load(run, "autoresearch_frozen_transfer_domctl.json")["results"][0]
        raw = res["combined"]["ece"]
        cal = res["combined_calibrated"]["ece"]
        base = res["base_calibrated"]["ece"]
        ax.plot([raw, cal], [y, y], color=INK_MUTED, linewidth=1.4, zorder=2,
                solid_capstyle="round")
        ax.plot([raw], [y], marker="o", markersize=6, color=C_ORANGE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.plot([cal], [y], marker="o", markersize=6, color=C_BLUE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.plot([base], [y], marker="|", markersize=8, color=INK,
                markeredgewidth=1.4, zorder=3)
        ax.annotate(f"{raw:.3f}", (raw, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.5, color=INK_2)
        ax.annotate(f"{cal:.3f}", (cal, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.5, color=INK_2)

    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _ in SETTINGS])
    ax.set_xlabel("expected calibration error (frozen test, P2)")
    ax.set_xlim(left=0.0)
    tidy(ax, grid_axis="x")
    ax.set_ylim(-0.6, len(SETTINGS) - 0.4)

    handles = [
        plt.Line2D([], [], color=C_ORANGE, marker="o", markersize=6,
                   linestyle="none", label="combined, uncalibrated"),
        plt.Line2D([], [], color=C_BLUE, marker="o", markersize=6,
                   linestyle="none", label="combined, Platt-calibrated"),
        plt.Line2D([], [], color=INK, marker="|", markersize=8,
                   linestyle="none", label="baseline, calibrated"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=2)

    out = OUTDIR / "fig_calibration.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 6 -- decision relevance vs draft cost
# --------------------------------------------------------------------------
def fig_regret() -> None:
    """Regret reduction from the pre-round signal as draft cost varies.

    Plotted as reduction (higher is better), i.e. the negation of the recorded
    ``delta_regret`` (which is combined minus base, so negative means helps).
    Filled markers are the costs where the recorded CI clears zero
    (``helps_ci``); hollow markers are where it does not. The honest shape of
    the result is that the signal is inert at cheap draft cost and only pays at
    expensive draft cost.
    """
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    markers = ["o", "s", "^"]

    for (label, run), color, marker in zip(SETTINGS, colors, markers):
        res = load(run, "autoresearch_frozen_transfer_domctl.json")["results"][0]
        sweep = res["regret_cost_sweep"]
        xs = [p["cost_draft"] for p in sweep]
        ys = [-p["delta_regret"] for p in sweep]
        ax.plot(xs, ys, color=color, linewidth=1.6, label=label, zorder=3)
        for p, x, y in zip(sweep, xs, ys):
            ax.plot([x], [y], marker=marker, markersize=4.6, color=color,
                    markerfacecolor=color if p["helps_ci"] else "white",
                    markeredgecolor=color, markeredgewidth=1.1, zorder=4)

    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 9.0])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4", "9"])
    ax.set_xlabel(r"draft cost ($\times$ verify)")
    ax.set_ylabel("regret reduction")
    tidy(ax, grid_axis="both")
    ax.legend(loc="upper left")

    handles = [
        plt.Line2D([], [], color=INK_2, marker="o", markersize=4.6,
                   linestyle="none", label="CI clears zero"),
        plt.Line2D([], [], color=INK_2, marker="o", markersize=4.6,
                   markerfacecolor="white", linestyle="none", label="CI includes zero"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", title=None)
    ax.add_artist(leg)
    series = [plt.Line2D([], [], color=c, marker=m, markersize=4.6, linewidth=1.6,
                         label=l) for (l, _), c, m in zip(SETTINGS, colors, markers)]
    ax.legend(handles=series, loc="upper left")

    out = OUTDIR / "fig_regret.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure 7 -- timing schematic (no artifacts; a diagram, not a plot)
# --------------------------------------------------------------------------
def fig_schematic() -> None:
    """Where the pre-round signal sits relative to every other control signal.

    Drawn here rather than in TikZ so the whole figure set builds from one
    toolchain and can be rendered and inspected without a LaTeX picture stack.
    """
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def box(x, w, y, h, label, face, edge, text_color=INK, fontsize=7.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                   linewidth=1.0, joinstyle="round", zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, zorder=4, linespacing=1.4)

    # Round r-1 verification, then round r.
    box(2, 20, 20, 8, "verify round $r{-}1$\n(target forward)", "#e8f1fd", C_BLUE)
    box(30, 18, 20, 8, "draft round $r$\n(drafter forward)", "#fdeee7", C_ORANGE)
    box(56, 18, 20, 8, "verify round $r$\n(target forward)", "#e8f1fd", C_BLUE)
    box(80, 18, 20, 8, "commit\n+ bonus token", "#f2f2ef", INK_MUTED)

    for x0, x1 in ((22, 30), (48, 56), (74, 80)):
        ax.annotate("", xy=(x1, 24), xytext=(x0, 24),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, linewidth=1.0))

    # The frontier state becomes available the instant r-1 verification ends.
    ax.plot([22, 22], [20, 13], color=C_BLUE, linewidth=1.2, linestyle="--",
            zorder=2)
    box(2, 38, 6, 7, r"$\phi(t)$ frontier state is already in memory"
                     "\n" r"$\bf{this\ work}$: pre-round probe",
        "#e8f1fd", C_BLUE, fontsize=7)

    # Everything else can only fire once draft tokens exist.
    ax.plot([48, 48], [20, 13], color=C_ORANGE, linewidth=1.2, linestyle="--",
            zorder=2)
    box(44, 55, 6, 7, "draft entropy / margin $\\cdot$ agreement heads on draft states"
                      "\nverification-time judges $\\cdot$ bandits on history",
        "#fdeee7", C_ORANGE, fontsize=7)

    ax.text(21, 2.5, "no draft compute spent", ha="center", fontsize=7,
            color=INK_2, style="italic")
    ax.text(71.5, 2.5, "draft compute already spent", ha="center", fontsize=7,
            color=INK_2, style="italic")

    ax.text(1, 31.5, "time", fontsize=7, color=INK_MUTED)
    ax.annotate("", xy=(99, 31.2), xytext=(8, 31.2),
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, linewidth=0.7))

    out = OUTDIR / "fig_schematic.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_schematic()
    fig_atlas()
    fig_dose()
    fig_length()
    fig_forest()
    fig_calibration()
    fig_regret()


if __name__ == "__main__":
    main()
