"""International Standard Atmosphere (ISA) model, troposphere only.

Provides air density, pressure, temperature, and speed of sound as a function
of geometric altitude. Good from sea level up to 11 km; above that the
temperature is held constant (isothermal layer approximation) so the model
never misbehaves at silly altitudes.

Reference: ISA sea-level values T0 = 288.15 K, p0 = 101325 Pa,
lapse rate L = 0.0065 K/m, R = 287.05 J/(kg K), gamma = 1.4.
"""

import math

T0 = 288.15        # K, sea-level temperature
P0 = 101325.0      # Pa, sea-level pressure
RHO0 = 1.225       # kg/m^3, sea-level density
LAPSE = 0.0065     # K/m, tropospheric lapse rate
R_AIR = 287.05     # J/(kg K)
GAMMA = 1.4
G0 = 9.80665
TROPOPAUSE = 11000.0  # m


class Atmosphere:
    """Stateless ISA atmosphere. All methods take altitude in meters."""

    def temperature(self, altitude_m: float) -> float:
        """Air temperature [K]."""
        h = max(0.0, altitude_m)
        if h <= TROPOPAUSE:
            return T0 - LAPSE * h
        return T0 - LAPSE * TROPOPAUSE  # isothermal above tropopause

    def pressure(self, altitude_m: float) -> float:
        """Air pressure [Pa]."""
        h = max(0.0, altitude_m)
        if h <= TROPOPAUSE:
            T = self.temperature(h)
            return P0 * (T / T0) ** (G0 / (R_AIR * LAPSE))
        # Isothermal layer decay above the tropopause.
        T_tp = T0 - LAPSE * TROPOPAUSE
        p_tp = P0 * (T_tp / T0) ** (G0 / (R_AIR * LAPSE))
        return p_tp * math.exp(-G0 * (h - TROPOPAUSE) / (R_AIR * T_tp))

    def density(self, altitude_m: float) -> float:
        """Air density [kg/m^3] from the ideal gas law."""
        T = self.temperature(altitude_m)
        return self.pressure(altitude_m) / (R_AIR * T)

    def speed_of_sound(self, altitude_m: float) -> float:
        """Speed of sound [m/s]."""
        return math.sqrt(GAMMA * R_AIR * self.temperature(altitude_m))
