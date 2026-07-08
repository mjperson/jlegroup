"""jlegroup — stellar-occultation light-curve forward modeling (MIT Elliot-group lineage).

Python implementations of the Elliot-group occultation methods:

    CE97 — Chamberlain & Elliot (1997), PASP 109, 1170.
           "A Numerical Method for Calculating Stellar Occultation Light Curves
           from an Arbitrary Atmospheric Model."  (this release)
    EY92 — Elliot & Young (1992), AJ 103, 991.
           "Analysis of Stellar Occultation Data for Planetary Atmospheres.
            I. Model Fitting, with Application to Pluto."  (this release)
    EPQ03 — Elliot, Person & Qu (2003), AJ 126, 1041.
           "Analysis of Stellar Occultation Data. II. Inversion, with
            Application to Pluto and Triton."  (this release)

Naming convention: the all-lowercase ``jlegroup`` is this Python package; the
camelCase ``jleGroup`` refers to the original Mathematica package family.
"""

__version__ = "0.3.0"

from . import CE97
from . import EPQ03
from . import EY92
from . import physicalData

__all__ = ["CE97", "EPQ03", "EY92", "physicalData", "__version__"]
