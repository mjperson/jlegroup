"""jlegroup — stellar-occultation light-curve forward modeling (MIT Elliot-group lineage).

Python implementations of the Elliot-group occultation methods:

    CE97 — Chamberlain & Elliot (1997), PASP 109, 1170.
           "A Numerical Method for Calculating Stellar Occultation Light Curves
           from an Arbitrary Atmospheric Model."  (this release)
    EY92 — Elliot & Young (1992), AJ 103, 991.
           "Analysis of Stellar Occultation Data for Planetary Atmospheres.
            I. Model Fitting, with Application to Pluto."  (this release)

Naming convention: the all-lowercase ``jlegroup`` is this Python package; the
camelCase ``jleGroup`` refers to the original Mathematica package family.
"""

__version__ = "0.2.0"

from . import CE97
from . import EY92
from . import physicalData

__all__ = ["CE97", "EY92", "physicalData", "__version__"]
