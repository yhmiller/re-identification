"""Derivation-consistent generalisation. The engineered contribution.

One modification to a standard de-identification pipeline, stated in full:

    When features are derived from a variable that has been generalised, derive
    them from the generalised values, not from the originals.

That is the whole change. It is a [MODIFY] operation on the release step of an
otherwise conventional generalisation pipeline. Nothing about the banding, the
suppression, the risk metrics or the utility model differs between the baseline
and the proposed method.

Why it matters
--------------
Custodians generalise a grade sequence to protect students, then publish the
engineered features their analysts asked for alongside it: the minimum, the
maximum, the mean, the trend. Those features are computed from the original
values and released at original precision.

They are not summaries in any protective sense. `gpa_min` and `gpa_max` are
exact order statistics of the sequence that was just banded, and each mean fixes
a sum across it. Together they constrain the banded values back towards the
numbers the bands were meant to hide, which `attack_models.reconstruct`
demonstrates by interval propagation.

Measured across three corpora, publishing them from the originals reverses 67%
to 100% of the protection the generalisation provided, and recovers 21% to 94%
of individual grades exactly. Recomputing them from the bands holds that
protection, and it has a price: it costs 2.7 to 13.5 AUC-PR points depending on
the corpus. The baseline release is therefore not dominated. It sits on the
frontier at the high-utility, low-protection end, and the contribution is the
measured trade-off rather than a free fix.

An earlier version of this docstring claimed the opposite, that utility was
"identical to three decimal places" and the baseline was a dominated choice.
That was an artefact of passing the utility model only the generalised source
columns, which are byte-identical between the two arms, so the model never saw
the one thing that differs between them. The claim was withdrawn from four
documents in August 2026. This docstring was missed at the time and is corrected
here. See the comment in `risk_utility.py` guarding the fixed code path.

Modes
-----
`NONE`      publish the generalised sequence, no derived features
`BASELINE`  derive from the originals, publish at original precision. This is
            what pipelines do today, and it is the arm the ablation reverts to
`PROPOSED`  derive from the generalised values. The contribution
"""

from dataclasses import dataclass, field

import pandas as pd

import deidentify as di

NONE = "none"
BASELINE = "baseline"
PROPOSED = "proposed"

MODES = (NONE, BASELINE, PROPOSED)

# The findings document and the earlier notebooks used descriptive names for the
# two derived modes. Kept so older result files remain readable.
ALIASES = {"leaky": BASELINE, "safe": PROPOSED}


@dataclass
class Release:
    """A file as a custodian would actually hand it out."""

    frame: pd.DataFrame
    mode: str
    band_width: float
    suppress_k: int
    records_suppressed: int
    source_columns: list
    derived_columns: list = field(default_factory=list)

    @property
    def visible_columns(self):
        """Everything the recipient can match on. Risk is measured over this."""
        return list(self.source_columns) + list(self.derived_columns)

    @property
    def fully_suppressed(self):
        """Suppression that empties the file is deletion, not protection."""
        return self.records_suppressed >= len(self.frame)


def build(frame, source_columns, band_width=None, suppress_k=None, mode=PROPOSED):
    """Produce one release.

    Algorithm, with the modified step marked:

        1. Generalise each source column to the given band width.
        2. Suppress records whose equivalence class falls below suppress_k.
        3. Derive the engineered features.
           [MODIFY] from the generalised frame under PROPOSED,
                    from the original frame under BASELINE.
        4. Join and return.

    Only step 3 differs between the two arms. Steps 1, 2 and 4 are identical, so
    any difference in disclosure risk is attributable to the derivation source
    and nothing else.
    """
    mode = ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    released = frame.copy()
    suppressed = 0

    if band_width:
        released = di.generalise(released, source_columns, band_width)

    if suppress_k and suppress_k > 1:
        released, suppressed = di.suppress(released, source_columns, suppress_k)

    derived_columns = []
    if mode != NONE:
        origin = frame if mode == BASELINE else released
        derived = di.derive_features(origin, source_columns)
        released = released.drop(columns=derived.columns, errors="ignore").join(derived)
        derived_columns = list(derived.columns)

    return Release(
        frame=released,
        mode=mode,
        band_width=band_width or 0.0,
        suppress_k=suppress_k or 0,
        records_suppressed=suppressed,
        source_columns=list(source_columns),
        derived_columns=derived_columns,
    )


def baseline(frame, source_columns, band_width=None, suppress_k=None):
    """Standard generalisation: derived features computed from the originals."""
    return build(frame, source_columns, band_width, suppress_k, mode=BASELINE)


def proposed(frame, source_columns, band_width=None, suppress_k=None):
    """Derivation-consistent generalisation: derived from the protected values."""
    return build(frame, source_columns, band_width, suppress_k, mode=PROPOSED)
