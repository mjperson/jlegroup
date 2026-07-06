"""Physical constants for jlegroup, mirroring the Mathematica ``jleGroup`physicalData``
package.

Vintage matters: the Mathematica package — and therefore every reference light curve used
to validate this code — uses CODATA-1986 values. Do NOT "upgrade" these to current CODATA
without revalidating the benchmark suite; a relative change of a few 1e-4 in a constant is
detectable at the accuracy this package is validated to.

Mapping to the Mathematica package:
    BOLTZMANN   = pdBoltzmannConstant[2]
    LOSCHMIDT   = pdLoschmidtNumber[3]
    refractivitySTP("N2", lambda) ~ pdRefractivityAtSTP["N2", 1, lambda]
"""

# --- CODATA 1986 vintage (jleGroup`physicalData) ------------------------------------
BOLTZMANN = 1.380658e-23        # J/K
GRAVITATIONAL = 6.67259e-11     # m^3 kg^-1 s^-2
AVOGADRO = 6.0221367e23         # mol^-1
LOSCHMIDT = 2.686763e25         # m^-3 (number density at STP)
AU_KM = 1.49597870691e8         # km per astronomical unit

# molar masses, kg/mol
MOLAR_MASS = {
    "N2": 28.01e-3,
}


def refractivitySTP(gas="N2", wavelength_um=0.7):
    """Refractivity (n - 1) of the gas at STP.

    N2: Peck & Khanna 1966 (JOSA 56, 8), 15 C dispersion formula scaled to 273.16 K.
    Valid roughly 0.47-2.1 um. At 0.7 um returns ~2.970e-4.

    wavelength_um: wavelength in microns.
    """
    if gas != "N2":
        raise NotImplementedError("refractivitySTP: only N2 is implemented so far")
    sigma = (1.0 / wavelength_um) ** 2
    return (273.16 + 15.0) / 273.16 * (6497.378 + 3.0738649e6 / (144.0 - sigma)) / 1e8


def refractivity(number_density, gas="N2", wavelength_um=0.7):
    """Refractivity profile nu(r) from a number-density profile [m^-3]:

        nu(r) = nu_STP(gas, lambda) * n(r) / LOSCHMIDT
    """
    return refractivitySTP(gas, wavelength_um) / LOSCHMIDT * number_density
