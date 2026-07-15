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
    shadowmap — occultation shadow maps (functionality of the Mathematica
           ``jleGroup`shadowMap`` package; no reference publication).
           Optional: requires the ``astropy`` extra —
           ``pip install "jlegroup[shadowmap]"`` — and loads lazily so the
           light-curve modules stay astropy-free.

Naming convention: the all-lowercase ``jlegroup`` is this Python package; the
camelCase ``jleGroup`` refers to the original Mathematica package family.
"""

__version__ = "0.9.0"

from . import CE97
from . import EPQ03
from . import EY92
from . import physicalData

__all__ = ["CE97", "EPQ03", "EY92", "physicalData", "shadowmap", "__version__"]


def __getattr__(name):
    # PEP 562: load shadowmap (and its astropy dependency) only on first use.
    # NB: must use importlib, not "from . import shadowmap" — the from-import
    # machinery probes the package with hasattr(), which re-enters this
    # __getattr__ and recurses.
    if name == "shadowmap":
        import importlib

        module = importlib.import_module(".shadowmap", __name__)
        globals()[name] = module  # cache so later access skips __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
