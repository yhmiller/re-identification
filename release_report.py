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
    metrics = am.recovery_metrics(
        lo, hi, summary, band_width, truth=frame[sequence]
    )
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
