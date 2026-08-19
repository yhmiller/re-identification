"""Figures for the manuscript.

One function per figure, each returning the path it wrote. Four figures, each
carrying one story:

    mechanism    what the derived features do to the protection
    frontier     the decision a custodian actually faces
    stability    whether the effect survives looking at individual folds
    explanation  the whole argument, for a reader who is not a privacy specialist

Conventions follow the Q1 results template. Baseline is dashed blue-grey,
proposed is solid green, values are labelled on the marks so a reader never has
to estimate against an axis, and no caption appears inside an image. Captions
belong below the figure in the manuscript, where they can be edited.

Everything is drawn from the result tables in results/disclosure/ rather than
recomputed, so a figure can never disagree with the number in the text.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "disclosure"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DPI = 300
BASELINE = "#5B7C99"
PROPOSED = "#2E7D4F"
NEUTRAL = "#9AA5AD"
GRID = "#DDE1E4"

CORPUS_LABEL = {
    "allied_health": "Allied health (n=110)",
    "nursing": "Nursing (n=566)",
    "public": "Public, replication (n=649)",
}
MARKER = {"allied_health": "o", "nursing": "s", "public": "^"}


def _style(ax):
    """House style. No seaborn, no default matplotlib grid."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#6B7075")
    ax.tick_params(colors="#3A3F44", labelsize=9)
    return ax


def _save(fig, name):
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_mechanism():
    """What the derived features do to the protection generalisation provides.

    Four states per corpus, read left to right: unprotected, generalised,
    generalised with baseline-derived features published alongside, and
    generalised with the features recomputed from the protected values.

    The point of the figure is the third bar. Generalisation lowers uniqueness;
    publishing features derived from the originals puts it back.
    """
    frame = pd.read_csv(RESULTS / "risk_utility_frontier.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)

    for ax, corpus in zip(axes, CORPUS_LABEL):
        sub = frame[frame["stratum"] == corpus]
        mid = sorted(sub[sub["band_width"] > 0]["band_width"].unique())[1]

        def pick(mode, width):
            row = sub[(sub["derived_mode"] == mode) & (sub["band_width"] == width)]
            return float(row["prop_unique"].iloc[0]) if len(row) else np.nan

        states = [
            ("Unprotected", pick("none", 0.0), NEUTRAL),
            ("Generalised", pick("none", mid), PROPOSED),
            ("+ baseline\nderivation", pick("baseline", mid), BASELINE),
            ("+ proposed\nderivation", pick("proposed", mid), PROPOSED),
        ]
        _style(ax)
        for i, (label, value, colour) in enumerate(states):
            ax.bar(
                i,
                value,
                color=colour,
                width=0.66,
                zorder=3,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(
                i,
                value + 0.025,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#3A3F44",
            )
        ax.set_xticks(range(4))
        ax.set_xticklabels([s[0] for s in states], fontsize=8, rotation=20,
                           ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(
            f"{CORPUS_LABEL[corpus]}\nband {mid:g}", fontsize=9.5, color="#3A3F44"
        )

    axes[0].set_ylabel("Records uniquely identifiable", fontsize=9.5)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _save(fig, "figure_a_mechanism")


def figure_frontier():
    """Risk reduction against utility retained. The decision figure.

    Every configuration is a point. Up and to the right is better on both axes,
    so the Pareto set is the upper-right boundary. Baseline-derivation points
    sit far left: they retain utility and buy almost no protection.
    """
    frame = pd.read_csv(RESULTS / "risk_utility_frontier.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _style(ax)

    for corpus, group in frame.groupby("stratum"):
        base = group[group["config"] == "original, no protection"].iloc[0]
        if base["prop_unique"] == 0:
            continue
        risk_cut = (base["prop_unique"] - group["prop_unique"]) / base["prop_unique"]
        utility = group["auc_pr"] / base["auc_pr"]

        for mode, colour, face in (
            ("baseline", BASELINE, BASELINE),
            ("proposed", PROPOSED, PROPOSED),
            ("none", NEUTRAL, "white"),
        ):
            mask = (group["derived_mode"] == mode).to_numpy()
            if not mask.any():
                continue
            ax.scatter(
                risk_cut[mask] * 100,
                utility[mask] * 100,
                marker=MARKER[corpus],
                s=64,
                zorder=4,
                facecolor=face,
                edgecolor=colour,
                linewidth=1.4,
                label=f"{CORPUS_LABEL[corpus].split(' (')[0]}, {mode}",
            )

    ax.axhline(100, color=NEUTRAL, linewidth=0.9, linestyle=":", zorder=2)
    ax.text(1, 100.4, "utility of the unprotected release", fontsize=8, color="#6B7075")
    ax.set_xlabel("Disclosure risk reduction (%)", fontsize=10)
    ax.set_ylabel("Analytical utility retained (% of unprotected AUC-PR)", fontsize=10)

    # Two small legends rather than one of nine combined entries: shape carries
    # the corpus, fill carries the derivation mode. A reader decodes each axis
    # of the encoding once instead of memorising nine labels.
    shape_keys = [plt.Line2D([], [], marker=MARKER[c], linestyle="", color="#6B7075",
                             markerfacecolor="white", markersize=7,
                             label=CORPUS_LABEL[c].split(" (")[0])
                  for c in CORPUS_LABEL]
    fill_keys = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=BASELINE, color=BASELINE, label="baseline derivation"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor=PROPOSED, color=PROPOSED, label="proposed derivation"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor="white", color=NEUTRAL, label="no derived features"),
    ]
    first = ax.legend(handles=shape_keys, fontsize=8, frameon=False,
                      loc="lower left", title="Corpus", title_fontsize=8.5)
    first._legend_box.align = "left"
    ax.add_artist(first)
    second = ax.legend(handles=fill_keys, fontsize=8, frameon=False,
                       loc="lower center", title="Release", title_fontsize=8.5)
    second._legend_box.align = "left"
    return _save(fig, "figure_b_frontier")


def figure_stability():
    """Distribution of the paired per-fold utility differences.

    Shows that the effect is not one convenient average, and shows equally that
    the allied health folds straddle zero. A figure reporting only means would
    conceal the one corpus that does not support the claim.
    """
    folds = pd.read_csv(RESULTS / "confirmatory_fold_differences.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    _style(ax)

    positions, labels, colours = [], [], []
    data = []
    pos = 0
    for corpus in CORPUS_LABEL:
        sub = folds[folds["stratum"] == corpus]
        for width in sorted(sub["band_width"].unique()):
            values = sub[sub["band_width"] == width]["difference"].to_numpy()
            data.append(values)
            positions.append(pos)
            labels.append(f"{width:g}")
            crosses_zero = values.min() < 0 < values.max()
            colours.append(BASELINE if crosses_zero else PROPOSED)
            pos += 1
        pos += 0.8

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.4),
        flierprops=dict(marker="", alpha=0),
        zorder=3,
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(7)
    for values, p in zip(data, positions):
        ax.scatter(
            p + rng.uniform(-0.16, 0.16, len(values)),
            values,
            s=9,
            color="#33383C",
            alpha=0.55,
            zorder=5,
            linewidth=0,
        )

    ax.axhline(0, color="#C0392B", linewidth=1.1, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Generalisation band width, grouped by corpus", fontsize=10)
    ax.set_ylabel("Per-fold AUC-PR difference\n(proposed minus baseline)", fontsize=10)

    centres = []
    i = 0
    for corpus in CORPUS_LABEL:
        n = folds[folds["stratum"] == corpus]["band_width"].nunique()
        centres.append((corpus, np.mean(positions[i : i + n])))
        i += n
    for corpus, centre in centres:
        ax.text(
            centre,
            ax.get_ylim()[1] * 0.97,
            CORPUS_LABEL[corpus].split(" (")[0],
            ha="center",
            fontsize=9,
            color="#3A3F44",
        )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=PROPOSED,
            label="all folds below zero",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=BASELINE,
            label="folds straddle zero",
        ),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    return _save(fig, "figure_c_stability")


def figure_explanation():
    """The argument as a flow, for a reader who is not a privacy specialist.

    Deliberately states the mechanism as a consistency, not a proof. The arrows
    say what was observed to happen, not that the derived features are proven to
    carry identical information to their sources.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, size=9):
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.3, zorder=3
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color="#22262A",
            zorder=4,
            linespacing=1.4,
        )

    def arrow(x1, y1, x2, y2, colour="#6B7075", label=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.4),
        )
        if label:
            ax.text(
                (x1 + x2) / 2 + 0.15,
                (y1 + y2) / 2,
                label,
                fontsize=8,
                color=colour,
                ha="left",
                va="center",
            )

    box(3.4, 8.6, 3.2, 1.0, "Original grade sequence", "#F2F4F5", NEUTRAL)
    arrow(5.0, 8.6, 5.0, 7.9)
    box(3.4, 6.9, 3.2, 1.0, "Generalise to bands\nrisk falls", "#E8F1EC", PROPOSED)

    arrow(4.2, 6.9, 2.4, 6.1)
    arrow(5.8, 6.9, 7.6, 6.1)

    box(
        0.5,
        4.9,
        3.6,
        1.2,
        "Derive features from\nthe ORIGINAL values",
        "#EAF0F4",
        BASELINE,
    )
    box(
        5.9,
        4.9,
        3.6,
        1.2,
        "Derive features from\nthe PROTECTED values",
        "#E8F1EC",
        PROPOSED,
    )

    arrow(2.3, 4.9, 2.3, 4.1, BASELINE)
    arrow(7.7, 4.9, 7.7, 4.1, PROPOSED)

    box(
        0.5,
        2.6,
        3.6,
        1.5,
        "Risk returns to near its\nunprotected level\n\nUtility also returns",
        "#F7ECEC",
        "#C0392B",
    )
    box(
        5.9,
        2.6,
        3.6,
        1.5,
        "Risk stays reduced\n\nUtility falls by a\nmeasured amount",
        "#E8F1EC",
        PROPOSED,
    )

    ax.text(
        5.0,
        1.6,
        "The same pathway carries both the disclosure risk and the analytical value.",
        ha="center",
        fontsize=9.5,
        color="#22262A",
    )
    ax.text(
        5.0,
        1.0,
        "Results are consistent with derived features computed from unprotected values\n"
        "acting as an information pathway that substantially reverses the effect of generalisation.",
        ha="center",
        fontsize=8.2,
        color="#6B7075",
        linespacing=1.5,
    )
    return _save(fig, "figure_d_explanation")


def build_all():
    return [
        figure_mechanism(),
        figure_frontier(),
        figure_stability(),
        figure_explanation(),
    ]


if __name__ == "__main__":
    for path in build_all():
        print(f"wrote {path.relative_to(ROOT)}")
