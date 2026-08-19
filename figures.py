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


# ---------------------------------------------------------------------------
# Methods figures. These describe the procedure rather than plotting a result,
# so they are drawn rather than read from a table.
# ---------------------------------------------------------------------------

INK = "#22262A"
MUTED = "#6B7075"


def _box(ax, x, y, w, h, text, face, edge, size=8.5, weight="normal"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                               linewidth=1.3, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=INK, zorder=4, linespacing=1.45, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, colour=MUTED, style="-|>", width=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=colour, linewidth=width))


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def figure_architecture():
    """Figure 1. The four phases, end to end."""
    fig, ax = _canvas(8.6, 6.4)

    _box(ax, 0.3, 8.3, 4.4, 1.1,
         "Ghanaian health professions records\nallied health, n=110   nursing, n=566",
         "#F2F4F5", NEUTRAL)
    _box(ax, 5.3, 8.3, 4.4, 1.1,
         "Public replication corpus\nUCI Student Performance, n=649",
         "#FAFAFA", NEUTRAL)

    for x in (2.5, 7.5):
        _arrow(ax, x, 8.3, x, 7.5)

    _box(ax, 0.3, 6.4, 9.4, 1.1,
         "Phase 1   Corpus construction and de-identification\n"
         "names discarded, index numbers replaced by keyed digests",
         "#EFF3F6", BASELINE)
    _arrow(ax, 5.0, 6.4, 5.0, 5.6)

    _box(ax, 0.3, 4.5, 9.4, 1.1,
         "Phase 2   Release construction\n"
         "generalise source variables, derive features from one of two sources",
         "#E8F1EC", PROPOSED)
    _arrow(ax, 2.6, 4.5, 2.6, 3.7)
    _arrow(ax, 7.4, 4.5, 7.4, 3.7)

    _box(ax, 0.3, 2.6, 4.4, 1.1,
         "Phase 3   Disclosure risk\nuniqueness, attacks",
         "#EFF3F6", BASELINE)
    _box(ax, 5.3, 2.6, 4.4, 1.1,
         "Phase 4   Analytical utility\npredictive performance",
         "#EFF3F6", BASELINE)

    _arrow(ax, 2.5, 2.6, 4.4, 1.9)
    _arrow(ax, 7.5, 2.6, 5.6, 1.9)
    _box(ax, 2.6, 0.7, 4.8, 1.1,
         "Risk-utility characterisation\nacross release configurations",
         "#E8F1EC", PROPOSED, weight="bold")
    return _save(fig, "figure_1_architecture")


def figure_replication():
    """Figure 2. Replication across corpora. There is no fusion point."""
    fig, ax = _canvas(8.8, 5.6)

    corpora = [
        (0.3, "Allied health\nn = 110\n6 positions, 0 to 4\ncredit-weighted",
         "study population", BASELINE, "#EFF3F6"),
        (3.5, "Nursing\nn = 566\n6 positions, 0 to 4\nunweighted",
         "study population", BASELINE, "#EFF3F6"),
        (6.7, "Public\nn = 649\n3 positions, 0 to 20\nperiod grades",
         "replication corpus", NEUTRAL, "#FAFAFA"),
    ]
    for x, text, role, edge, face in corpora:
        _box(ax, x, 7.2, 3.0, 1.9, text, face, edge, size=8)
        ax.text(x + 1.5, 6.95, role, ha="center", fontsize=7.5, color=MUTED,
                style="italic")

    for x in (1.8, 5.0, 8.2):
        _arrow(ax, x, 6.75, x, 5.9)
        _box(ax, x - 1.5, 4.6, 3.0, 1.3,
             "Identical procedure\napplied independently", "#E8F1EC", PROPOSED,
             size=8)
        _arrow(ax, x, 4.6, x, 3.3)
        _box(ax, x - 1.5, 2.0, 3.0, 1.3, "Results reported\nseparately",
             "#F2F4F5", NEUTRAL, size=8)

    ax.add_patch(plt.Rectangle((0.3, 0.35), 9.4, 1.15, facecolor="#FBF3F3",
                               edgecolor="#C0392B", linewidth=1.2, zorder=3))
    ax.text(5.0, 0.92,
            "No fusion point. The corpora are never merged: grade scales, grade-point\n"
            "definitions and sequence lengths differ, so a pooled corpus would attribute to the\n"
            "procedure differences that arise from the measurement scales.",
            ha="center", va="center", fontsize=8, color=INK, zorder=4,
            linespacing=1.5)
    return _save(fig, "figure_2_replication")


def figure_baseline_vs_proposed():
    """Figure 3. The two release constructions, with the modified step marked."""
    fig, ax = _canvas(9.2, 5.0)

    _box(ax, 3.4, 8.6, 3.2, 0.9, "Source sequence  X", "#F2F4F5", NEUTRAL, size=9)
    _arrow(ax, 4.4, 8.6, 3.4, 7.3)
    _arrow(ax, 5.6, 8.6, 6.6, 7.3)

    ax.text(2.6, 7.35, "Baseline", ha="center", fontsize=10.5, color=BASELINE,
            fontweight="bold")
    ax.text(7.4, 7.35, "Proposed", ha="center", fontsize=10.5, color=PROPOSED,
            fontweight="bold")

    _box(ax, 0.4, 5.9, 4.4, 1.2, "Generalise\ng(X)", "#EFF3F6", BASELINE)
    _box(ax, 5.2, 5.9, 4.4, 1.2, "Generalise\ng(X)", "#E8F1EC", PROPOSED)

    _arrow(ax, 2.6, 5.9, 2.6, 4.9, BASELINE)
    _arrow(ax, 7.4, 5.9, 7.4, 4.9, PROPOSED)

    _box(ax, 0.4, 3.5, 4.4, 1.4, "Derive features from\nthe ORIGINAL values\n d(X)",
         "#EFF3F6", BASELINE)
    _box(ax, 5.2, 3.5, 4.4, 1.4,
         "Derive features from\nthe GENERALISED values\n d(g(X))", "#E8F1EC", PROPOSED)

    ax.annotate("", xy=(5.2, 4.2), xytext=(4.8, 4.2),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", linewidth=2.0))
    ax.text(5.0, 4.62, "[M]", ha="center", fontsize=10, color="#C0392B",
            fontweight="bold")
    ax.text(5.0, 3.02, "the single modified step", ha="center", fontsize=8,
            color="#C0392B", style="italic")

    _arrow(ax, 2.6, 3.5, 2.6, 2.5, BASELINE)
    _arrow(ax, 7.4, 3.5, 7.4, 2.5, PROPOSED)
    _box(ax, 0.4, 1.3, 4.4, 1.2, "R_baseline = [ g(X) | d(X) ]", "#EFF3F6", BASELINE, size=9)
    _box(ax, 5.2, 1.3, 4.4, 1.2, "R_proposed = [ g(X) | d(g(X)) ]", "#E8F1EC", PROPOSED, size=9)

    ax.text(5.0, 0.65,
            "Both releases publish the same generalised columns and the same feature names.",
            ha="center", fontsize=8.5, color=MUTED)
    return _save(fig, "figure_3_baseline_vs_proposed")


ALGORITHM_1 = """Algorithm 1: Derivation-consistent generalisation

Input:  source sequence X (n x p), band width w, suppression threshold k,
        derivation function d, derivation source mode m in {baseline, proposed}
Output: release R

1:  G <- generalise(X, w)                      band each source value
2:  G, n_suppressed <- suppress(G, k)          blank classes below k
3:  S <- X if m = baseline else G              <- [M] THE MODIFIED STEP
4:  D <- d(S)                                  derive the published features
5:  R <- [ G | D ]                             join and release
6:  return R

Only line 3 differs between the two arms. Lines 1, 2, 4 and 5 are identical,
so any measured difference is attributable to the derivation source alone."""


def algorithm_box():
    """Algorithm 1, rendered to match the manuscript's figure conventions.

    The text is also carried in manuscript/methods.md, which is authoritative
    and editable. This image exists so the layout matches the other figures.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#FBFCFC", edgecolor=BASELINE,
                               linewidth=1.4, zorder=1))
    for i, line in enumerate(ALGORITHM_1.split("\n")):
        colour = "#C0392B" if "[M]" in line else INK
        weight = "bold" if line.startswith("Algorithm") or "[M]" in line else "normal"
        ax.text(0.03, 0.94 - i * 0.062, line, transform=ax.transAxes,
                fontsize=8.2, family="monospace", color=colour,
                fontweight=weight, va="top", zorder=3)
    return _save(fig, "algorithm_1")


def table_1_corpora():
    """Table 1. Corpus descriptives, generated from the corpora themselves.

    Written as CSV for the record and as Markdown for the manuscript, so the
    numbers in the text cannot drift from the numbers in the data.
    """
    import strata as st

    rows = []
    for name, s in st.load_all().items():
        seq = s.frame[s.sequence]
        observed = seq.notna().sum(axis=1)
        rows.append({
            "Corpus": name.replace("_", " "),
            "Role": "replication" if name == "public" else "study population",
            "Students": s.n,
            "Grade scale": "0 to 4" if s.scale_max == 4 else "0 to 20",
            "Aggregate": ("credit-weighted CGPA" if name == "allied_health"
                          else "unweighted mean" if name == "nursing" else "none"),
            "Sequence positions": len(s.sequence),
            "Positions observed, median": int(observed.median()),
            "Quasi-identifiers": len(s.quasi_identifiers()),
            "Demographics": len(s.demographics),
            "Band widths": ", ".join(f"{w:g}" for w in s.band_widths),
            "Sensitive prevalence": f"{s.frame[st.SENSITIVE].mean():.1%}",
        })

    frame = pd.DataFrame(rows).set_index("Corpus").T
    frame.to_csv(RESULTS / "table_1_corpora.csv")

    lines = ["| | " + " | ".join(frame.columns) + " |",
             "|---" * (len(frame.columns) + 1) + "|"]
    for label, row in frame.iterrows():
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    path = RESULTS / "table_1_corpora.md"
    path.write_text("\n".join(lines) + "\n")
    return path
