"""Figures and tables for the manuscript.

One function per item, each returning the path it wrote. `build_all` produces
every one in manuscript order, so the whole set regenerates with one command and
no image can drift out of step with the table it came from.

Methods items describe the procedure and are drawn:

    architecture           Figure 1, the four phases end to end
    replication            Figure 2, three corpora, no fusion point
    baseline_vs_proposed   Figure 3, the two releases with the modified step marked
    algorithm_box          Algorithm 1, the procedure, modification at line 3
    table_1_corpora        corpus descriptives, generated from the corpora

Results items plot measurements and are read from the result tables:

    mechanism    Figure 4, what the derived features do to the protection
    frontier     Figure 5, the decision a custodian actually faces
    stability    Figure 6, whether the effect survives looking at individual folds
    explanation  Figure 7, the whole argument for a non-specialist reader.
                 Belongs to the Discussion, not the Results

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
    return _save(fig, "figure_s1_mechanism")


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
    return _save(fig, "figure_7_frontier")


def figure_fold_differences():
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
    return _save(fig, "figure_s2_fold_differences")


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

    # Two-line caption. The vertical gap is wider than it looks it needs to be
    # because the second block is itself two lines; at 1.6 and 1.0 they collided.
    ax.text(
        5.0,
        1.7,
        "The same pathway carries both the disclosure risk and the analytical value.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#22262A",
    )
    ax.text(
        5.0,
        0.75,
        "Results are consistent with derived features computed from unprotected values\n"
        "acting as an information pathway that substantially reverses the effect of generalisation.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#6B7075",
        linespacing=1.5,
    )
    return _save(fig, "figure_10_explanation")


def build_all():
    """Every figure and table, in manuscript order."""
    return [
        figure_conceptual_framework(),
        figure_architecture(),
        figure_replication(),
        figure_baseline_vs_proposed(),
        algorithm_box(),
        table_1_corpora(),
        # Results tables, generated so the manuscript never transcribes them.
        table_10_main_comparison(),
        table_11_column_survival(),
        table_12_statistical(),
        table_13_reconstruction(),
        table_14_error_exchange(),
        # Results, Figures 5 to 9, in manuscript order.
        figure_pr_curves(),
        figure_fold_distribution(),
        figure_frontier(),
        figure_importance(),
        figure_confusion(),
        # Discussion.
        figure_explanation(),
        # Supplementary. Displaced from the main section by the eight-item cap,
        # not by being uninformative.
        figure_mechanism(),
        figure_fold_differences(),
    ]




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
    return _save(fig, "figure_2_architecture")


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
    return _save(fig, "figure_3_replication")


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
    return _save(fig, "figure_4_baseline_vs_proposed")


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


if __name__ == "__main__":
    for path in build_all():
        print(f"wrote {path.relative_to(ROOT)}")


def figure_conceptual_framework():
    """The conceptual framework, for the Introduction.

    Distinct from Figure 2, the study architecture, and from Figure 8, which
    explains the mechanism once the results are in. This one states the
    relationships the study sets out to measure, before any measurement.

    Read top to bottom: a source variable is generalised, features are derived,
    and the pair becomes the release a recipient holds. The release is then read
    on two axes at once. The marked edge is the only one a custodian controls,
    and it is what the study prices.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, size=8.5, weight="normal"):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                          linewidth=1.3, zorder=3)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=INK, zorder=4, linespacing=1.45,
                fontweight=weight)

    box(3.2, 8.5, 3.6, 1.1, "Source variable\ngrade sequence", "#EEF1F3", MUTED)

    box(3.2, 6.4, 3.6, 1.2, "GENERALISATION\nband to width w", "#EEF1F3", INK,
        weight="bold")
    _arrow(ax, 5.0, 8.5, 5.0, 7.6)

    box(3.2, 4.3, 3.6, 1.2, "DERIVATION\nd(.) to 8 features", "#EEF1F3", INK,
        weight="bold")
    _arrow(ax, 5.0, 6.4, 5.0, 5.5)

    # The one edge a custodian controls, and the study's whole subject.
    ax.text(7.15, 4.9, "derived from the\nORIGINAL values\nor the PROTECTED\nvalues",
            ha="left", va="center", fontsize=8, color="#B5484A",
            linespacing=1.4, fontweight="bold")
    _arrow(ax, 6.8, 4.9, 7.05, 4.9, colour="#B5484A")

    box(2.9, 2.4, 4.2, 1.1, "RELEASE\nwhat the recipient holds", "#EEF1F3",
        MUTED, weight="bold")
    _arrow(ax, 5.0, 4.3, 5.0, 3.5)

    # The release is read on two axes at once. That simultaneity is the premise.
    box(0.2, 0.5, 3.6, 1.1, "Disclosure risk\nuniqueness, k", "#FBEDED", "#B5484A")
    box(6.2, 0.5, 3.6, 1.1, "Analytical value\nAUC-PR", "#EAF2EC", PROPOSED)
    _arrow(ax, 4.2, 2.4, 2.6, 1.6)
    _arrow(ax, 5.8, 2.4, 7.4, 1.6)

    ax.text(5.0, 1.05, "measured\ntogether", ha="center", va="center",
            fontsize=8, color=MUTED, linespacing=1.4, style="italic")
    return _save(fig, "figure_1_conceptual_framework")


def figure_importance():
    """Figure 9. What the model relies on, under each derivation arm.

    Results R6 asks where the engineered component ranks against the native
    columns, per dataset, never averaged. The engineered component here is not a
    feature but the source the derived block is computed from, so the question
    becomes how much of the model's reliance the derived block carries and
    whether changing its source moves that.

    Two bars per corpus rather than a conventional importance plot, because the
    finding is the shift between arms rather than any single ranking.
    """
    table = pd.read_csv(RESULTS / "interpretability_summary.csv")
    order = ["allied_health", "nursing", "public"]
    labels = {"allied_health": "Allied health", "nursing": "Nursing",
              "public": "Public"}

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width, positions = 0.36, np.arange(len(order))

    for offset, mode, colour, name in (
        (-width / 2, "baseline", BASELINE, "derived from original values"),
        (width / 2, "proposed", PROPOSED, "derived from protected values"),
    ):
        shares = [
            float(table[(table.stratum == s)
                        & (table.derived_mode == mode)]
                  .derived_share_of_importance.iloc[0]) * 100
            for s in order
        ]
        bars = ax.bar(positions + offset, shares, width, label=name,
                      color=colour, edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, shares):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                    f"{value:.0f}%", ha="center", fontsize=8.5, color=INK)

    ax.set_xticks(positions)
    ax.set_xticklabels([labels[s] for s in order])
    ax.set_ylabel("Share of model importance carried\nby the derived features (%)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "figure_8_importance")



def _modelled_label(corpus, errors):
    """Corpus label carrying the number of records actually modelled.

    CORPUS_LABEL quotes the corpus size. Listwise deletion on the predictor set
    means the utility model sees fewer, so a curve or a confusion matrix titled
    with the corpus size overstates its own denominator.
    """
    base = CORPUS_LABEL[corpus].split(" (")[0]
    row = errors[errors.stratum == corpus]
    if row.empty:
        return CORPUS_LABEL[corpus]
    return f"{base} (n={int(row.iloc[0].n_records)})"


def figure_pr_curves():
    """R2. Precision-recall curves for both arms, one panel per corpus.

    Curves come from the averaged out-of-fold probabilities of stage 10, so the
    curve and the AUC-PR in Table 5 describe the same predictions. The no-skill
    floor is drawn on each panel because a precision-recall curve read against
    zero rather than against prevalence overstates every model on it.
    """
    curves = pd.read_csv(RESULTS / "pr_curves.csv")
    summary = pd.read_csv(RESULTS / "fold_summary.csv")
    errors = pd.read_csv(RESULTS / "error_profile.csv")

    corpora = [c for c in CORPUS_LABEL if c in set(curves.stratum)]
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.0 * len(corpora), 3.8))
    axes = np.atleast_1d(axes)

    for ax, corpus in zip(axes, corpora):
        _style(ax)
        for mode, colour, style in (
            ("baseline", BASELINE, "--"),
            ("proposed", PROPOSED, "-"),
        ):
            sub = curves[(curves.stratum == corpus) & (curves.derived_mode == mode)]
            score = summary[
                (summary.stratum == corpus) & (summary.derived_mode == mode)
            ]
            label = mode.capitalize()
            if not score.empty:
                label += f" (AUC-PR {score.iloc[0].auc_pr_mean:.3f})"
            ax.plot(sub.recall, sub.precision, style, color=colour,
                    linewidth=1.8, label=label, zorder=3)

        row = errors[errors.stratum == corpus]
        if not row.empty:
            floor = row.iloc[0].support_positive / row.iloc[0].n_records
            ax.axhline(floor, color=NEUTRAL, linewidth=1.0, linestyle=":",
                       zorder=2)
            ax.text(0.02, floor + 0.02, f"no-skill floor {floor:.3f}",
                    fontsize=7.5, color="#5A6066")

        ax.set_title(_modelled_label(corpus, errors), fontsize=9.5,
                     color="#3A3F44")
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    axes[0].set_ylabel("Precision", fontsize=9)
    fig.tight_layout()
    return _save(fig, "figure_5_pr_curves")


def figure_fold_distribution():
    """R3. Per-fold AUC-PR for each arm, with every fold drawn.

    The template asks for the metric itself per arm rather than the paired
    difference, so a reader can see the two distributions and their overlap.
    The paired differences, which are what the confirmatory test operates on,
    are the supplementary fold-difference figure.
    """
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    errors = pd.read_csv(RESULTS / "error_profile.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    _style(ax)

    rng = np.random.default_rng(ru_seed())
    positions, labels, data, colours = [], [], [], []
    pos = 0
    for corpus in CORPUS_LABEL:
        sub = folds[folds.stratum == corpus]
        if sub.empty:
            continue
        for mode, colour in (("baseline", BASELINE), ("proposed", PROPOSED)):
            values = sub[sub.derived_mode == mode].auc_pr.to_numpy()
            data.append(values)
            positions.append(pos)
            labels.append(mode.capitalize())
            colours.append(colour)
            pos += 1
        pos += 0.9

    box = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                     medianprops=dict(color="white", linewidth=1.4),
                     flierprops=dict(marker="", linestyle="none"), zorder=2)
    for patch, colour in zip(box["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
        patch.set_edgecolor(colour)

    # Every fold drawn, jittered so overlapping folds remain countable.
    for values, position in zip(data, positions):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.plot(position + jitter, values, "o", markersize=2.6,
                color="#2B2F33", alpha=0.65, zorder=4)

    for values, position in zip(data, positions):
        ax.text(position, values.max() + 0.028,
                f"{values.mean():.3f}\n±{values.std(ddof=1):.3f}",
                ha="center", fontsize=7, color="#3A3F44")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUC-PR per fold", fontsize=9)
    ax.set_ylim(min(v.min() for v in data) - 0.06,
                max(v.max() for v in data) + 0.10)

    centres = [np.mean(positions[i:i + 2]) for i in range(0, len(positions), 2)]
    present = [c for c in CORPUS_LABEL if c in set(folds.stratum)]
    for centre, corpus in zip(centres, present):
        ax.text(centre, ax.get_ylim()[0] + 0.012,
                _modelled_label(corpus, errors),
                ha="center", fontsize=8.5, color="#3A3F44")

    fig.tight_layout()
    return _save(fig, "figure_6_fold_distribution")


def figure_confusion():
    """R7. Confusion matrices for both arms, normalised by true class.

    Normalised by row, so each cell is the share of an actual class that landed
    in a predicted class and the two corpora of different size can be read on
    the same scale. Raw counts are printed beneath each proportion, since a
    proportion over a small class hides how few records produced it.
    """
    errors = pd.read_csv(RESULTS / "error_profile.csv")
    corpora = [c for c in CORPUS_LABEL if c in set(errors.stratum)]

    fig, axes = plt.subplots(2, len(corpora),
                             figsize=(2.9 * len(corpora), 5.6))
    axes = np.atleast_2d(axes)

    for column, corpus in enumerate(corpora):
        for row_index, mode in enumerate(("baseline", "proposed")):
            ax = axes[row_index][column]
            row = errors[(errors.stratum == corpus)
                         & (errors.derived_mode == mode)]
            if row.empty:
                ax.axis("off")
                continue
            r = row.iloc[0]
            counts = np.array([[r.true_negative, r.false_positive],
                               [r.false_negative, r.true_positive]], dtype=float)
            shares = counts / counts.sum(axis=1, keepdims=True)

            ax.imshow(shares, cmap="Greens", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{shares[i, j]:.2f}\n({int(counts[i, j])})",
                            ha="center", va="center", fontsize=8.5,
                            color="white" if shares[i, j] > 0.55 else "#2B2F33")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Not weak", "Weak"], fontsize=7.5)
            ax.set_yticklabels(["Not weak", "Weak"], fontsize=7.5)
            ax.grid(False)
            if row_index == 0:
                ax.set_title(_modelled_label(corpus, errors), fontsize=9,
                             color="#3A3F44")
            if column == 0:
                ax.set_ylabel(f"{mode.capitalize()}\nActual", fontsize=8.5)
            if row_index == 1:
                ax.set_xlabel("Predicted", fontsize=8.5)

    fig.tight_layout()
    return _save(fig, "figure_9_confusion")


def ru_seed():
    """The figure jitter must not move between runs."""
    import risk_utility
    return risk_utility.BASE_SEED


def table_10_main_comparison():
    """R2's main comparison table, generated from the frontier.

    Written out rather than typed into the manuscript. The AUC-ROC column of an
    earlier hand-built version of this table was wrong in every row, which is
    what generating it prevents.
    """
    frontier = pd.read_csv(RESULTS / "risk_utility_frontier.csv")
    bands = {"allied_health": 0.5, "nursing": 0.5, "public": 2.5}
    order = [
        ("original, no protection", "Unprotected"),
        ("none", "Generalised, no derived features"),
        ("baseline", "Generalised + baseline derivation"),
        ("proposed", "Generalised + proposed derivation"),
    ]

    lines = [
        "| Corpus | Release | Unique | Δ unique | AUC-PR (mean ± SD) | Δ AUC-PR | AUC-ROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for corpus, band in bands.items():
        rows = {}
        for _, row in frontier[frontier.stratum == corpus].iterrows():
            if row.config == "original, no protection":
                rows["original, no protection"] = row
            elif row.suppress_k == 0 and row.band_width == band:
                rows[row.derived_mode] = row

        baseline, proposed = rows.get("baseline"), rows.get("proposed")
        label = CORPUS_LABEL[corpus].split(" (")[0]
        for key, description in order:
            row = rows.get(key)
            if row is None:
                continue
            delta_u = delta_p = ""
            if key == "proposed" and baseline is not None:
                delta_u = (
                    f"**{(proposed.prop_unique - baseline.prop_unique) * 100:+.1f} pp**"
                )
                delta_p = f"**{proposed.auc_pr - baseline.auc_pr:+.3f}**"
            lines.append(
                f"| {label if key == order[0][0] else ''} | {description} "
                f"| {row.prop_unique * 100:.1f}% | {delta_u} "
                f"| {row.auc_pr:.3f} ± {row.auc_pr_sd:.3f} | {delta_p} "
                f"| {row.auc_roc:.3f} |"
            )

    path = RESULTS / "table_10_main_comparison.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def table_12_statistical():
    """R4's confirmatory comparison table, generated from the two stat files.

    Risk and utility comparisons live in separate outputs and were previously
    merged by hand into the manuscript, which is where five of the nine effect
    sizes went wrong.
    """
    risk = pd.read_csv(RESULTS / "confirmatory_risk.csv")
    utility = pd.read_csv(RESULTS / "confirmatory_utility.csv")
    merged = risk.merge(utility, on=["stratum", "band_width"],
                        suffixes=("_risk", "_utility"))

    def interval(low, high):
        return f"[{low:+.3f}, {high:+.3f}]".replace("+0.000", "0.000")

    lines = [
        "| Corpus | Band | Risk difference | 95% CI | Utility difference "
        "| 95% CI, corrected | p, corrected | d_z |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for corpus in CORPUS_LABEL:
        sub = merged[merged.stratum == corpus].sort_values("band_width")
        label = CORPUS_LABEL[corpus].split(" (")[0]
        for position, row in enumerate(sub.itertuples()):
            p = row.corrected_p
            printed = "<0.0001" if p < 0.0001 else f"{p:.4f}".rstrip("0")
            lines.append(
                f"| {label if position == 0 else ''} | {row.band_width:.2f} "
                f"| {row.difference_full_corpus:+.3f} "
                f"| {interval(row.ci_low, row.ci_high)} "
                f"| {row.mean_difference:+.3f} "
                f"| {interval(row.corrected_ci_low, row.corrected_ci_high)} "
                f"| {printed} | {row.cohens_d:+.2f} |"
            )

    path = RESULTS / "table_12_statistical.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def table_13_reconstruction():
    """R5's reconstruction table, generated, with both risk quantities shown.

    The hand-built predecessor carried a column headed "After reversion" holding
    `prop_unique_after_attack`, which is what an adversary recovers from
    reconstructed point estimates. A reader took it for the baseline release's
    own uniqueness, which is a different number, and four apparent
    cross-table discrepancies followed. Both quantities are now printed side by
    side under names that distinguish them.
    """
    recon = pd.read_csv(RESULTS / "experiment_b_reconstruction.csv")
    risk = pd.read_csv(RESULTS / "confirmatory_risk.csv")
    merged = recon.merge(
        risk[["stratum", "band_width", "prop_unique_baseline"]],
        on=["stratum", "band_width"], how="left")

    lines = [
        "| Corpus | Band | Unprotected | Generalised, no derived block "
        "| Baseline release, as published | Recovered by the attack "
        "| Protection reversed | Source values recovered exactly |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for corpus in CORPUS_LABEL:
        sub = merged[merged.stratum == corpus].sort_values("band_width")
        label = CORPUS_LABEL[corpus].split(" (")[0]
        for position, row in enumerate(sub.itertuples()):
            lines.append(
                f"| {label if position == 0 else ''} | {row.band_width:.2f} "
                f"| {row.prop_unique_original * 100:.1f}% "
                f"| {row.prop_unique_after_generalisation * 100:.1f}% "
                f"| {row.prop_unique_baseline * 100:.1f}% "
                f"| {row.prop_unique_after_attack * 100:.1f}% "
                f"| {row.protection_reversed * 100:.1f}% "
                f"| {row.prop_values_recovered_exactly * 100:.1f}% |"
            )
    path = RESULTS / "table_13_reconstruction.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def table_14_error_exchange():
    """R7's error-profile exchange, in stakeholder units.

    Methods M15 declares the false negative the costlier error and the false
    positive a review that finds nothing. This table applies that declaration to
    the measured counts, which is the translation the metric alone does not make.
    """
    errors = pd.read_csv(RESULTS / "error_profile.csv")
    lines = [
        "| Corpus | False negatives, baseline → proposed | Struggling students "
        "additionally reached | False positives, baseline → proposed "
        "| Extra reviews finding nothing | Reviews per additional student reached |",
        "|---|---|---|---|---|---|",
    ]
    for corpus in CORPUS_LABEL:
        sub = errors[errors.stratum == corpus].set_index("derived_mode")
        if not {"baseline", "proposed"} <= set(sub.index):
            continue
        base, prop = sub.loc["baseline"], sub.loc["proposed"]
        reached = int(base.false_negative - prop.false_negative)
        extra = int(prop.false_positive - base.false_positive)
        ratio = f"{extra / reached:.1f}" if reached else "not defined"
        lines.append(
            f"| {CORPUS_LABEL[corpus].split(' (')[0]} "
            f"| {int(base.false_negative)} → {int(prop.false_negative)} | {reached} "
            f"| {int(base.false_positive)} → {int(prop.false_positive)} | {extra} "
            f"| {ratio} |")
    path = RESULTS / "table_14_error_exchange.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def table_11_column_survival():
    """Per-arm derived-column survival, the GATE-2 condition.

    The single-distinct-value filter of M7 holds the feature set constant across
    folds but cannot hold it constant across arms, because a column derived from
    banded values can collapse where the same column at full precision does not.
    Whether it actually did is a fact, and this is the fact.
    """
    frontier = pd.read_csv(RESULTS / "risk_utility_frontier.csv")
    arms = frontier[(frontier.suppress_k == 0)
                    & (frontier.derived_mode.isin(["baseline", "proposed"]))]
    lines = ["| Corpus | Band | Source columns | Derived columns, baseline "
             "| Derived columns, proposed | Identical |",
             "|---|---|---|---|---|---|"]
    for corpus in CORPUS_LABEL:
        sub = arms[arms.stratum == corpus]
        source = frontier[(frontier.stratum == corpus)
                          & (frontier.derived_mode == "none")].n_features.iloc[0]
        label = CORPUS_LABEL[corpus].split(" (")[0]
        for position, band in enumerate(sorted(sub.band_width.unique())):
            row = sub[sub.band_width == band]
            b = int(row[row.derived_mode == "baseline"].n_features.iloc[0]) - int(source)
            p = int(row[row.derived_mode == "proposed"].n_features.iloc[0]) - int(source)
            lines.append(
                f"| {label if position == 0 else ''} | {band:.2f} | {int(source)} "
                f"| {b} of 8 | {p} of 8 | {'yes' if b == p else '**no**'} |")
    path = RESULTS / "table_11_column_survival.md"
    path.write_text("\n".join(lines) + "\n")
    return path
