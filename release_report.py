"""One release configuration in, the five numbers a custodian needs out.

The writing plan, Stage 8, asks for exactly this:

    "A single function that turns a release configuration into the numbers a
    custodian needs: risk, drivers, leakage, retained utility, recommendation
    against a threshold. The notebooks already compute all five separately."

They did, in five places. This assembles them behind one call so the app has
something to talk to, and so the assembly itself is testable rather than living
inside a user interface.

Aggregate by construction
-------------------------
Every field returned is a scalar or a per-attribute summary. Nothing here is
per-record, and nothing here can be made per-record without changing the return
type, which a test asserts. That is deliberate. This study's whole argument is
that per-student output derived from academic records is re-identifying, so a
custodian-facing tool that emitted it would demonstrate the harm the thesis
documents.

The `drivers` table is one row per *attribute*, not per student. An attribute is
a column name, which is not personal data.
"""

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import attack_models as am
import deidentify as di
import derivation_consistent as dc
import disclosure_risk as dr
import risk_utility as ru

# Illustrative defaults, not standards. A custodian brings their own tolerance;
# these exist so the report can say something rather than nothing. The risk
# threshold is expressed as the share of records that may be uniquely
# identifiable. The utility tolerance is in AUC-PR points.
DEFAULT_RISK_THRESHOLD = 0.05
DEFAULT_UTILITY_TOLERANCE = 0.02

VERDICT_RELEASE = "within tolerance"
VERDICT_REVIEW = "review before release"
VERDICT_HOLD = "do not release as configured"


@dataclass
class ReleaseReport:
    """What a custodian needs to decide, and nothing that identifies anyone."""

    # Configuration
    corpus: str
    n_records: int
    band_width: float
    derived_mode: str
    suppress_k: int

    # 1. Risk
    prop_unique: float
    min_class_size: int
    prosecutor_risk: float
    marketer_risk: float
    share_below_k5: float

    # 2. Leakage: what publishing from originals instead would restore
    prop_unique_if_derived_from_originals: float
    leakage: float
    values_pinned_exactly: float

    # 3. Utility
    auc_pr: float
    auc_pr_unprotected: float
    auc_pr_retained: float
    prevalence: float

    # 4. Recommendation
    risk_threshold: float
    utility_tolerance: float
    verdict: str
    reasons: list

    # 5. Drivers, one row per attribute
    drivers: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def as_row(self):
        """Flat scalars only, for writing to a table."""
        row = {k: v for k, v in asdict(self).items() if k != "drivers"}
        row["reasons"] = "; ".join(self.reasons)
        return row


def _leakage(frame, sequence, band_width, protected_unique):
    """How much of the protection publishing from originals would give back.

    Runs the baseline arm and compares its uniqueness against the protected
    arm's, as a share of what generalisation removed.

    This is **not** the same quantity Experiment B reports as
    `protection_reversed`, and the two do not agree. Experiment B routes through
    the reconstruction attack and asks what uniqueness an attacker recovers.
    This routes through the release itself and asks how many records are unique
    in the file as published. On the public corpus at band 2.5 they give 87.8%
    and 91.9% respectively.

    The release route is used here because it is the one a custodian can act on:
    it answers "if I publish it the other way, how many of my records are
    singled out", without assuming an attacker does any work at all.
    """
    unprotected = dr.risk_profile(frame, sequence).prop_unique
    baseline = dc.baseline(frame, sequence, band_width=band_width)
    baseline_unique = dr.risk_profile(
        baseline.frame, baseline.visible_columns
    ).prop_unique

    removed = unprotected - protected_unique
    restored = baseline_unique - protected_unique
    return baseline_unique, (restored / removed if removed > 0 else 0.0)


def _values_pinned(frame, sequence, band_width):
    """Share of individual values an attacker could narrow to a single point.

    Runs the reconstruction procedure against the baseline arm, which is the
    configuration that leaks. Returns one aggregate proportion.
    """
    banded = di.generalise(frame, sequence, band_width)
    low, high = di.band_intervals(frame, sequence, band_width)
    derived = di.derive_features(frame, sequence)
    lo, hi, summary = am.reconstruct(banded, sequence, low, high, derived)
    # truth is the sequence columns only; passing the whole frame breaks on
    # any non-numeric column, which the public corpus has.
    metrics = am.recovery_metrics(lo, hi, summary, band_width, truth=frame[sequence])
    return float(metrics["prop_values_recovered_exactly"])


def _decide(prop_unique, retained, risk_threshold, utility_tolerance):
    """Turn the measurements into a verdict and the reasons behind it."""
    reasons = []
    if prop_unique > risk_threshold:
        reasons.append(
            f"{prop_unique:.1%} of records are uniquely identifiable, above the "
            f"{risk_threshold:.0%} tolerance set for this report"
        )
    if not np.isnan(retained) and (1.0 - retained) > utility_tolerance:
        reasons.append(
            f"{1 - retained:.1%} of analytical utility is lost, above the "
            f"{utility_tolerance:.0%} tolerance set for this report"
        )

    if not reasons:
        return VERDICT_RELEASE, ["both measures are inside the stated tolerances"]
    if len(reasons) == 1 and prop_unique <= risk_threshold:
        return VERDICT_REVIEW, reasons
    if prop_unique > risk_threshold:
        return VERDICT_HOLD, reasons
    return VERDICT_REVIEW, reasons


def build(
    frame,
    sequence,
    predictors,
    target,
    corpus="dataset",
    band_width=0.5,
    suppress_k=None,
    derived_mode=dc.PROPOSED,
    risk_threshold=DEFAULT_RISK_THRESHOLD,
    utility_tolerance=DEFAULT_UTILITY_TOLERANCE,
    with_utility=True,
):
    """Assemble the full report for one release configuration.

    `with_utility=False` skips the model fits, which are the slow part. The
    interface uses it to show risk immediately and fill in utility after.
    """
    release = dc.build(frame, sequence, band_width, suppress_k, derived_mode)
    profile = dr.risk_profile(release.frame, release.visible_columns)

    baseline_unique, leakage = _leakage(
        frame, sequence, band_width, profile.prop_unique
    )
    pinned = _values_pinned(frame, sequence, band_width)

    if with_utility:
        protected = ru.frontier_row(
            frame,
            sequence,
            predictors,
            target,
            "protected",
            band_width=band_width,
            suppress_k=suppress_k,
            derived_mode=derived_mode,
        )
        unprotected = ru.frontier_row(
            frame,
            sequence,
            predictors,
            target,
            "unprotected",
            derived_mode=dc.NONE,
        )
        auc_pr = protected["auc_pr"]
        auc_pr_unprotected = unprotected["auc_pr"]
        prevalence = protected["prevalence"]
        retained = (
            auc_pr / auc_pr_unprotected
            if auc_pr_unprotected and not np.isnan(auc_pr_unprotected)
            else np.nan
        )
    else:
        auc_pr = auc_pr_unprotected = prevalence = retained = np.nan

    verdict, reasons = _decide(
        profile.prop_unique, retained, risk_threshold, utility_tolerance
    )

    drivers = dr.solo_identifying_power(release.frame, release.visible_columns)
    drivers = drivers.sort_values("prop_unique_alone", ascending=False)

    return ReleaseReport(
        corpus=corpus,
        n_records=len(frame),
        band_width=float(band_width or 0.0),
        derived_mode=release.mode,
        suppress_k=int(suppress_k or 0),
        prop_unique=profile.prop_unique,
        min_class_size=profile.min_class_size,
        prosecutor_risk=profile.prosecutor_risk,
        marketer_risk=profile.marketer_risk,
        share_below_k5=profile.share_below_k.get(5, np.nan),
        prop_unique_if_derived_from_originals=baseline_unique,
        leakage=leakage,
        values_pinned_exactly=pinned,
        auc_pr=auc_pr,
        auc_pr_unprotected=auc_pr_unprotected,
        auc_pr_retained=retained,
        prevalence=prevalence,
        risk_threshold=risk_threshold,
        utility_tolerance=utility_tolerance,
        verdict=verdict,
        reasons=reasons,
        drivers=drivers,
    )


def plain_english(report):
    """The report as sentences, for a reader who does not read tables.

    Deliberately avoids "the model predicts" phrasing. Nothing here is about an
    individual, and the language should not suggest otherwise.
    """
    lines = [
        f"Of {report.n_records} records, "
        f"{report.prop_unique:.1%} would be uniquely identifiable in this release.",
        f"The smallest group of look-alike records contains "
        f"{report.min_class_size} record"
        f"{'s' if report.min_class_size != 1 else ''}.",
    ]

    if report.derived_mode == dc.PROPOSED:
        lines.append(
            f"Publishing the derived features from the original values instead "
            f"would raise that to "
            f"{report.prop_unique_if_derived_from_originals:.1%}, giving back "
            f"{report.leakage:.0%} of the protection this release provides."
        )
    else:
        lines.append(
            f"This release publishes derived features computed from the "
            f"original values, which gives back {report.leakage:.0%} of the "
            f"protection the generalisation provided."
        )

    lines.append(
        f"An attacker using only what is published could pin "
        f"{report.values_pinned_exactly:.0%} of individual source values exactly."
    )

    if not np.isnan(report.auc_pr_retained):
        lines.append(
            f"A model trained on this release reaches {report.auc_pr:.3f} AUC-PR "
            f"against {report.auc_pr_unprotected:.3f} unprotected, retaining "
            f"{report.auc_pr_retained:.0%} of the analytical value. Random "
            f"guessing on this task would score about {report.prevalence:.3f}."
        )

    lines.append(f"Verdict: {report.verdict}. " + "; ".join(report.reasons) + ".")
    return lines


def _is_dominated(candidate, others):
    """One configuration is dominated if another beats it on both axes.

    Same definition the study uses in notebook 07: another configuration cuts at
    least as much risk while keeping at least as much utility, and strictly beats
    it on one of the two. Dominated configurations are never the right choice, so
    what remains is the frontier a custodian picks from.
    """
    return any(
        other["risk_reduction"] >= candidate["risk_reduction"]
        and other["auc_pr_retained"] >= candidate["auc_pr_retained"]
        and (
            other["risk_reduction"] > candidate["risk_reduction"]
            or other["auc_pr_retained"] > candidate["auc_pr_retained"]
        )
        for other in others
        if other is not candidate
    )


def sweep(
    frame,
    sequence,
    predictors,
    target,
    corpus="dataset",
    band_widths=(),
    modes=(dc.PROPOSED, dc.BASELINE, dc.NONE),
    suppress_k=None,
    risk_threshold=DEFAULT_RISK_THRESHOLD,
    utility_tolerance=DEFAULT_UTILITY_TOLERANCE,
    with_utility=True,
    progress=None,
):
    """Every configuration for one dataset, with the frontier marked.

    A single verdict tells a custodian their release is unacceptable without
    telling them what to do instead. The contribution of this study is a
    trade-off with a menu, so the tool should show the menu.

    The unprotected release is measured once rather than once per configuration.
    `release_report.build` recomputes it every call, which is correct in
    isolation and wasteful across a sweep: it roughly halves the time here.

    `progress` is an optional callable taking (done, total) so a caller can
    report progress without this module importing an interface library.
    """
    rows = []
    unprotected_auc = np.nan
    prevalence = np.nan

    if with_utility:
        unprotected = ru.frontier_row(
            frame,
            sequence,
            predictors,
            target,
            "unprotected",
            derived_mode=dc.NONE,
        )
        unprotected_auc = unprotected["auc_pr"]
        prevalence = unprotected["prevalence"]

    total = len(band_widths) * len(modes)
    done = 0

    for band_width in band_widths:
        for mode in modes:
            release = dc.build(frame, sequence, band_width, suppress_k, mode)
            profile = dr.risk_profile(release.frame, release.visible_columns)

            if with_utility:
                protected = ru.frontier_row(
                    frame,
                    sequence,
                    predictors,
                    target,
                    "protected",
                    band_width=band_width,
                    suppress_k=suppress_k,
                    derived_mode=mode,
                )
                auc_pr = protected["auc_pr"]
                retained = (
                    auc_pr / unprotected_auc
                    if unprotected_auc and not np.isnan(unprotected_auc)
                    else np.nan
                )
            else:
                auc_pr = retained = np.nan

            verdict, reasons = _decide(
                profile.prop_unique, retained, risk_threshold, utility_tolerance
            )
            rows.append(
                {
                    "corpus": corpus,
                    "band_width": float(band_width),
                    "derived_mode": release.mode,
                    "prop_unique": profile.prop_unique,
                    "min_class_size": profile.min_class_size,
                    "auc_pr": auc_pr,
                    "auc_pr_retained": retained,
                    "verdict": verdict,
                    "reasons": "; ".join(reasons),
                }
            )

            done += 1
            if progress is not None:
                progress(done, total)

    table = pd.DataFrame(rows)

    # Risk reduction is measured against the unprotected release, so the two
    # axes point the same way: higher is better on both.
    unprotected_unique = dr.risk_profile(frame, sequence).prop_unique
    table["risk_reduction"] = (
        (unprotected_unique - table.prop_unique) / unprotected_unique
        if unprotected_unique > 0
        else 0.0
    )

    if with_utility and table.auc_pr_retained.notna().any():
        records = table.to_dict("records")
        table["on_frontier"] = [not _is_dominated(r, records) for r in records]
    else:
        table["on_frontier"] = False

    table.attrs["unprotected_auc_pr"] = unprotected_auc
    table.attrs["unprotected_prop_unique"] = unprotected_unique
    table.attrs["prevalence"] = prevalence
    return table


def recommend(table):
    """The frontier configuration that passes on both tolerances, if any.

    Prefers the largest risk reduction among acceptable options, because a
    custodian asking this question is protecting data, not maximising a model.
    Returns None when nothing on the frontier is acceptable, which is a real
    answer rather than a failure: on the smallest corpora, nothing is.
    """
    acceptable = table[(table.verdict == VERDICT_RELEASE) & table.on_frontier]
    if acceptable.empty:
        return None
    return acceptable.sort_values("risk_reduction", ascending=False).iloc[0]


REPORT_LIMITS = [
    "These are measured properties of this dataset under the attacks tested. "
    "They are not guarantees. An attacker holding information this assessment "
    "did not simulate could do better.",
    "Uniqueness is an upper bound on risk, not a probability of "
    "re-identification. An adversary with imprecise knowledge does "
    "considerably worse, by a wide margin in the corpora this method was "
    "developed on.",
    "The tolerances are values the assessor chose. They are not standards, and "
    "no regulator has endorsed them.",
    "Small cohorts cannot be protected by generalisation at any band width "
    "tested. Where this assessment refuses every configuration for a small "
    "group, that is the correct answer rather than a limitation of the tool.",
    "This assessment covers disclosure from the released file alone. It does "
    "not cover onward linkage to data the assessor does not hold.",
]


def to_markdown(report, sweep_table=None, assessed_by="", purpose="",
                recipient="", stamped=None):
    """The assessment as a document a custodian can attach to a decision.

    The writing plan calls Phase 6 "the practical contribution": converting the
    frontier into a written standard with worked examples. A report that can go
    into an ethics application or a data-sharing file is that worked example.

    `stamped` is passed in rather than read from the clock so the output is
    reproducible and so a caller can record when the assessment was run rather
    than when the document was rendered.

    Contains no per-record content, by the same rule as everything else here.
    The drivers table is one row per column and the frontier one row per
    configuration.
    """
    when = stamped or "not recorded"
    lines = [
        "# Release assessment",
        "",
        f"**Dataset** {report.corpus}, {report.n_records} records  ",
        f"**Assessed** {when}  ",
    ]
    if assessed_by:
        lines.append(f"**Assessed by** {assessed_by}  ")
    if purpose:
        lines.append(f"**Purpose of release** {purpose}  ")
    if recipient:
        lines.append(f"**Intended recipient** {recipient}  ")

    lines += [
        "",
        "## Verdict",
        "",
        f"**{report.verdict.upper()}**",
        "",
    ]
    lines += [f"- {reason}" for reason in report.reasons]

    lines += [
        "",
        "## Configuration assessed",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Generalisation band width | {report.band_width:g} |",
        f"| Derived features computed from | "
        f"{'the protected values' if report.derived_mode == dc.PROPOSED else 'the original values' if report.derived_mode == dc.BASELINE else 'not published'} |",
        f"| Suppression threshold | "
        f"{report.suppress_k if report.suppress_k else 'none'} |",
        f"| Risk tolerance applied | {report.risk_threshold:.0%} of records unique |",
        f"| Utility tolerance applied | {report.utility_tolerance:.0%} of value lost |",
        "",
        "## What this release discloses",
        "",
        "| Measure | Value | Meaning |",
        "|---|---|---|",
        f"| Uniquely identifiable | {report.prop_unique:.1%} | "
        "Share of records no other record matches |",
        f"| Smallest look-alike group | {report.min_class_size} | "
        "How many records the most exposed person hides among |",
        f"| Prosecutor risk | {report.prosecutor_risk:.3f} | "
        "Upper bound when the adversary knows the target is present |",
        f"| Marketer risk | {report.marketer_risk:.3f} | "
        "Mean risk across records, for an adversary tolerating error |",
        f"| Below k = 5 | {report.share_below_k5:.1%} | "
        "Share of records in groups smaller than five |",
        "",
        "## Leakage through derived features",
        "",
        f"Publishing the derived features from the original values would leave "
        f"**{report.prop_unique_if_derived_from_originals:.1%}** of records "
        f"uniquely identifiable, giving back **{report.leakage:.0%}** of the "
        f"protection generalisation provides.",
        "",
        f"Using only what is published, an attacker could narrow "
        f"**{report.values_pinned_exactly:.0%}** of individual source values to "
        f"a single point.",
        "",
    ]

    if not np.isnan(report.auc_pr_retained):
        lines += [
            "## Analytical value retained",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| AUC-PR on this release | {report.auc_pr:.3f} |",
            f"| AUC-PR unprotected | {report.auc_pr_unprotected:.3f} |",
            f"| Value retained | {report.auc_pr_retained:.0%} |",
            f"| No-skill floor for this task | {report.prevalence:.3f} |",
            "",
            "The floor is what guessing would score. A release is useful when "
            "its AUC-PR sits well above it, not merely above zero.",
            "",
        ]

    if len(report.drivers):
        lines += [
            "## What drives the risk",
            "",
            "One row per column, not per person. Read as: if someone knew only "
            "this column, how often would that alone single a record out.",
            "",
            "| Column | Distinct values | Singles out, alone |",
            "|---|---|---|",
        ]
        for row in report.drivers.head(10).itertuples():
            lines.append(
                f"| `{row.attribute}` | {row.n_distinct_values} | "
                f"{row.prop_unique_alone:.1%} |"
            )
        lines.append("")

    if sweep_table is not None and len(sweep_table):
        lines += [
            "## Options considered",
            "",
            "Every configuration measured the same way. An option is on the "
            "frontier when no other option beats it on both axes at once.",
            "",
            "| Band | Derived from | Unique | Risk removed | Value kept | "
            "Frontier | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in sweep_table.sort_values(
            ["band_width", "derived_mode"]
        ).itertuples():
            kept = (
                "n/a" if np.isnan(row.auc_pr_retained)
                else f"{row.auc_pr_retained:.0%}"
            )
            lines.append(
                f"| {row.band_width:g} | {row.derived_mode} | "
                f"{row.prop_unique:.1%} | {row.risk_reduction:.0%} | {kept} | "
                f"{'yes' if row.on_frontier else ''} | {row.verdict} |"
            )
        lines.append("")

    lines += ["## Limitations", ""]
    lines += [f"{n}. {text}" for n, text in enumerate(REPORT_LIMITS, 1)]

    lines += [
        "",
        "## Method",
        "",
        "Disclosure risk is measured as the share of records unique on the "
        "released columns, with the standard k-anonymity family reported "
        "alongside. Analytical value is measured by fitting a gradient-boosted "
        "tree model to exactly what a recipient receives and comparing its "
        "AUC-PR against the same model on the unprotected release.",
        "",
        "The single decision this assessment turns on is whether features "
        "derived from a generalised variable are computed from the original "
        "values or from the protected ones. Computing them from the protected "
        "values means a recipient can recompute them from the columns already "
        "published, so they disclose nothing further. Computing them from the "
        "originals means they constrain the generalised values back towards "
        "the numbers the generalisation was applied to hide.",
        "",
        "Produced by the Release Risk Advisor, from the MSc thesis "
        "*Derivation-Consistent De-identification of Health Professions "
        "Education Records: Disclosure Risk and Analytical Utility Under "
        "Hybrid Public and Institutional Academic Data* "
        "(Prince Bortey Miller, KNUST, supervised by Dr. Eric Opoku Osei).",
        "",
        "Decision support only. This document records an assessment; it does "
        "not authorise a release.",
        "",
    ]
    return "\n".join(lines)
