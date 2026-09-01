"""Derivation-consistent generalisation. The engineered contribution.

One modification to a standard de-identification pipeline:

    When features are derived from a variable that has been generalised, derive
    them from the generalised values, not from the originals.

It is a [MODIFY] operation on the release step alone. Banding, suppression, risk
metrics and the utility model are identical between arms, so any measured
difference is attributable to the derivation source.

Custodians generalise a grade sequence to protect students, then publish derived
features alongside it at original precision. Those features are not summaries in
any protective sense: `gpa_min` and `gpa_max` are exact order statistics of the
sequence just banded, and each mean fixes a sum across it, so together they
constrain the banded values back towards the numbers the bands were meant to
hide. `attack_models.reconstruct` demonstrates this by interval propagation.

Modes
-----
`NONE`      publish the generalised sequence, no derived features
`BASELINE`  derive from the originals, publish at original precision. This is
            what pipelines do today, and the arm the ablation reverts to
`PROPOSED`  derive from the generalised values
`SELECTIVE` derive only the arithmetically constraining features from the
            generalised values, leaving the rest at original precision

SELECTIVE exists to test whether the constraining classification in
`deidentify.CONSTRAINING_COLUMNS` yields a usable intermediate release. It need
not: under a mixed derivation source `gpa_trend` at original precision is the
exact difference between two source values, which combined with their published
bands pins the pair to a one-dimensional family. The containment and recovery
checks are run against this arm for that reason.
"""

from dataclasses import dataclass, field

import pandas as pd

import deidentify as di

NONE = "none"
BASELINE = "baseline"
PROPOSED = "proposed"
SELECTIVE = "selective"

MODES = (NONE, BASELINE, PROPOSED, SELECTIVE)

# Descriptive names retained so older result files remain readable.
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
                    from the original frame under BASELINE,
                    from both under SELECTIVE, split by Table 5's
                    constraining classification.
        4. Join and return.

    Only step 3 differs between arms.
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
    if mode == SELECTIVE:
        # Both blocks are computed over the same columns, so the published
        # schema stays identical to the other arms.
        protected = di.derive_features(released, source_columns)
        original = di.derive_features(frame, source_columns)
        free = [c for c in original.columns if c not in di.CONSTRAINING_COLUMNS]
        derived = protected[
            [c for c in protected.columns if c in di.CONSTRAINING_COLUMNS]
        ].join(original[free])
        derived = derived[list(original.columns)]
        released = released.drop(columns=derived.columns, errors="ignore").join(derived)
        derived_columns = list(derived.columns)
    elif mode != NONE:
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


def selective(frame, source_columns, band_width=None, suppress_k=None):
    """Constraining features from protected values, the rest from originals."""
    return build(frame, source_columns, band_width, suppress_k, mode=SELECTIVE)


def baseline(frame, source_columns, band_width=None, suppress_k=None):
    """Standard generalisation: derived features computed from the originals."""
    return build(frame, source_columns, band_width, suppress_k, mode=BASELINE)


def proposed(frame, source_columns, band_width=None, suppress_k=None):
    """Derivation-consistent generalisation: derived from the protected values."""
    return build(frame, source_columns, band_width, suppress_k, mode=PROPOSED)
